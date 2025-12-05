#!/usr/bin/env python
# video_processing_gui.py

import os
import re
import logging
import platform
import numpy as np
import cv2
import json
import ast  # ADDED: To handle single-quote JSON errors
import argparse
import subprocess
import gc
import warnings
from dotenv import load_dotenv
from openai import OpenAI

# --- Third Party Libraries ---
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import whisper
from panns_inference import AudioTagging, labels as pann_labels
import librosa
from moviepy import VideoFileClip
import imageio_ffmpeg 

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration & Logging ---
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

LOG_FILE = "video_processing.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# --- Path Setup ---
if platform.system() == "Windows":
    PANN_MODEL_PATH = r"C:\Users\naval\panns_data\cnn14.pth"
else:
    PANN_MODEL_PATH = "cnn14.pth" 

# --- YOLO Class Mapping ---
CLASS_MAP = {
    0: "person", 56: "chair", 57: "couch", 58: "potted plant", 
    60: "dining table", 62: "tv", 63: "laptop", 67: "cell phone"
}

# --- GPU Utilities ---
def free_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def seconds_to_timestr(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"

# --- Model Loading ---
def get_yolo_model():
    from ultralytics import YOLO
    logging.info("Loading YOLO model...")
    return YOLO("yolo11x.pt")

def get_blip_model():
    logging.info("Loading BLIP model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return model, processor, device

# --- Audio Functions ---
def extract_audio(video_path, audio_path):
    with VideoFileClip(video_path) as clip:
        clip.audio.write_audiofile(audio_path, logger=None)

def transcribe_audio(audio_file):
    logging.info("Loading Whisper (small)...")
    model = whisper.load_model("small") 
    try:
        result = model.transcribe(audio_file, language="en", condition_on_previous_text=False)
        return result["text"], result.get("segments", [])
    finally:
        del model
        free_gpu()

def detect_audio_events(audio_file):
    logging.info("Running PANNs detection...")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pt_path = PANN_MODEL_PATH
        if not os.path.exists(pt_path):
            logging.warning(f"PANNs checkpoint not found at {pt_path}. Attempting default load.")
            pt_path = None 

        model = AudioTagging(checkpoint_path=pt_path, device=device)
        waveform, sr = librosa.load(audio_file, sr=32000)
        
        events = {}
        segment_len = 5 * sr
        for i in range(0, len(waveform), segment_len):
            chunk = waveform[i:i+segment_len]
            if len(chunk) < sr: continue
            
            chunk_tensor = torch.tensor(chunk[None, :]).float().to(device)
            clipwise_output, _ = model.inference(chunk_tensor)
            
            clipwise_output = clipwise_output.cpu().detach().numpy()[0]
            if np.max(clipwise_output) > 0.2: 
                idx = np.argmax(clipwise_output)
                label = pann_labels[idx]
                time_str = seconds_to_timestr(i/sr)
                if label not in events: events[label] = []
                events[label].append(time_str)
        return events
    except Exception as e:
        logging.error(f"Audio Event Error: {e}")
        return {}
    finally:
        free_gpu()

# --- LLM Functions ---
def call_openai(prompt, model="gpt-4o"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional Video Editor."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return str(e)

def get_viral_prompt(report_text):
    return f"""
    Analyze the video report below.
    TASK: Identify 5-7 most engaging video segments for a "Highlight Reel".
    
    CRITICAL OUTPUT FORMAT:
    - Return ONLY valid JSON.
    - Use DOUBLE QUOTES (") for all keys and strings.
    - Do NOT use single quotes.
    - Format: List of dictionaries.
    
    Structure:
    [
        {{
            "start": "HH:MM:SS",
            "end": "HH:MM:SS", 
            "title": "Short Title",
            "description": "Overlay Text (Max 5 words)",
            "reasoning": "Why this clip matters"
        }}
    ]
    
    REPORT DATA:
    {report_text}
    """

def generate_video_descriptions(model_arg):
    """Generates viral_cuts.json with robust error handling."""
    if not os.path.exists("report.txt"):
        logging.error("report.txt not found.")
        return

    with open("report.txt", "r", encoding="utf-8") as f:
        report_text = f.read()

    model = "gpt-4o" 
    logging.info(f"Generating viral cuts using {model}...")
    
    response = call_openai(get_viral_prompt(report_text), model)
    
    # --- ROBUST JSON PARSING ---
    data = []
    try:
        # 1. Try to find the list content specifically
        start_index = response.find('[')
        end_index = response.rfind(']') + 1
        
        if start_index != -1 and end_index != -1:
            json_str = response[start_index:end_index]
            
            # 2. Try Standard JSON Load
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                logging.warning("JSON Decode failed. Trying Python Literal Eval (Single Quotes fix)...")
                # 3. Fallback: Python Literal Eval (Handles single quotes)
                try:
                    data = ast.literal_eval(json_str)
                except Exception as e:
                    logging.error(f"Failed to parse AI output: {e}")
                    logging.info(f"Raw Output: {json_str}")
                    return # Exit if we can't parse
            
            # 4. Save Bridge File (viral_cuts.json)
            with open("viral_cuts.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            logging.info("SUCCESS: 'viral_cuts.json' saved.")
            
            # 5. Save Readable Report (viral_report.txt)
            report_lines = ["--- VIRAL VIDEO PLAN ---", ""]
            for clip in data:
                report_lines.append(f"[{clip['start']} - {clip['end']}] {clip['title']}")
                report_lines.append(f"   Overlay: {clip['description']}")
                report_lines.append(f"   Why: {clip['reasoning']}\n")
                
            with open("viral_report.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(report_lines))
            logging.info("SUCCESS: 'viral_report.txt' saved.")
            
        else:
            logging.error("LLM did not return a list brackets [].")
            logging.info(f"Raw response: {response}")

    except Exception as e:
        logging.error(f"Critical Error in JSON Generation: {e}")

# --- Main Logic ---
def process_video(video_path, model_name, sample_rate=250):
    if not os.path.exists(video_path):
        logging.error(f"File {video_path} not found.")
        return

    # 1. Audio Processing
    audio_path = "temp_audio.wav"
    extract_audio(video_path, audio_path)
    
    transcript, segments = transcribe_audio(audio_path)
    audio_events = detect_audio_events(audio_path)
    
    if os.path.exists(audio_path): os.remove(audio_path)

    # 2. Visual Processing
    logging.info("Starting Visual Analysis...")
    yolo = get_yolo_model()
    blip_model, blip_proc, device = get_blip_model()
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = 0
    last_caption = ""
    
    report_data = []
    report_data.append(f"Video: {video_path}\n=== TRANSCRIPT ===")
    for s in segments:
        report_data.append(f"[{seconds_to_timestr(s['start'])} -> {seconds_to_timestr(s['end'])}] {s['text']}")
    
    report_data.append(f"\n=== AUDIO EVENTS ===\n{str(audio_events)}")
    report_data.append("\n=== VISUAL LOG ===")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        
        if frame_idx % sample_rate != 0: continue

        # Visual Chain
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        
        try:
            inputs = blip_proc(pil_img, return_tensors="pt").to(device)
            out = blip_model.generate(**inputs, max_length=50, min_length=10)
            caption = blip_proc.decode(out[0], skip_special_tokens=True)
        except:
            caption = ""

        # Filter & Deduplicate
        if not caption or len(caption) < 5: continue
        if caption.strip().lower() == last_caption.strip().lower(): continue
        last_caption = caption

        # YOLO (Only if scene is valid)
        yolo_res = yolo(frame, verbose=False)
        objects = []
        for r in yolo_res:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id in CLASS_MAP:
                    objects.append(CLASS_MAP[cls_id])
                else:
                    objects.append("object")
        
        unique_objs = list(set(objects))
        
        timestamp = seconds_to_timestr(frame_idx / fps)
        entry = f"Time {timestamp} | Scene: {caption} | Objects: {unique_objs}"
        logging.info(entry)
        report_data.append(entry)

    cap.release()
    free_gpu()

    # 3. Save Final Report
    with open("report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_data))
    
    # 4. Generate JSON Bridge
    generate_video_descriptions(model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--sample_rate", type=int, default=250)
    args = parser.parse_args()
    
    process_video(args.video, args.model, args.sample_rate)