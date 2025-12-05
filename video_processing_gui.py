#!/usr/bin/env python
# video_processing_cli.py

import os
import re
import logging
import platform
import psutil
import numpy as np
from ultralytics import YOLO
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import whisper
from panns_inference import AudioTagging, labels as pann_labels
import librosa
import soundfile as sf
from moviepy import VideoFileClip
import subprocess
import shutil
import warnings
import gc
import argparse
from dotenv import load_dotenv  # Loads .env file
from openai import OpenAI       # OpenAI Client
import cv2 # Moved import to top level for better practice

# --- Load Environment Variables ---
load_dotenv()

# --- Dynamic Path Setup ---
if platform.system() == "Windows":
    PANN_MODEL_PATH = r"C:\Users\naval\panns_data\cnn14.pth"
else:
    PANN_MODEL_PATH = "/app/models/pann/cnn14.pth"

# --- Suppress extraneous warnings ---
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

# --- Setup Logging ---
LOG_FILE = "video_processing.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# --- YOLO Class Mapping ---
CLASS_MAP = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
    50: "brocolli", 51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut",
    55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
    65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush"
}

# --- GPU MEMORY MANAGEMENT ---
global_yolo_model = None
global_blip_model = None
global_blip_processor = None

total_vram_gb = 0
if torch.cuda.is_available():
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

KEEP_MODELS_LOADED = total_vram_gb > 8.0
logging.info(f"VRAM Detected: {total_vram_gb:.2f} GB. Keep Models Loaded: {KEEP_MODELS_LOADED}")

def free_gpu(force=False):
    if force or not KEEP_MODELS_LOADED:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

def print_hardware_usage():
    print("=== Hardware Usage ===")
    print(f"CPU Usage: {psutil.cpu_percent()}%")
    mem = psutil.virtual_memory()
    print(f"Memory Usage: {mem.used / (1024 ** 3):.1f} GB / {mem.total / (1024 ** 3):.1f} GB ({mem.percent}%)")
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB")
    print("======================\n")

# --- Model Loading Helpers ---
def get_yolo_model():
    global global_yolo_model
    if global_yolo_model is None:
        logging.info("Loading YOLO model into memory...")
        global_yolo_model = YOLO("yolo11x.pt")
    return global_yolo_model

def get_blip_model():
    global global_blip_model, global_blip_processor
    if global_blip_model is None or global_blip_processor is None:
        logging.info("Loading BLIP model into memory...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        global_blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        global_blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        global_blip_model.to(device)
    return global_blip_model, global_blip_processor

def seconds_to_timestr(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"

# --- Audio Processing ---
def preprocess_audio(audio_file, sr=32000):
    waveform, sr = librosa.load(audio_file, sr=sr)
    if np.max(np.abs(waveform)) > 0:
        waveform = waveform / np.max(np.abs(waveform))
    return waveform, sr

def extract_audio(video_path, audio_path):
    with VideoFileClip(video_path) as clip:
        if clip.audio is None:
            raise ValueError("No audio track found in the video.")
        clip.audio.write_audiofile(audio_path, logger=None)

# --- UPDATED: Return segments for timeline ---
def transcribe_audio(audio_file, language="en"):  # Force default to English
    # REMOVED: waveform, sr = preprocess_audio(audio_file) 
    # Directly use the file extracted by MoviePy, it is cleaner.
    
    segments = [] 
    
    try:
        # Load the model with specific decoding options to reduce hallucinations
        options = dict(language="en", task="transcribe", condition_on_previous_text=False)
        
        # Use the raw audio_file directly
        result = whisper_model.transcribe(audio_file, **options)
        
        detected_language = "en" # We forced it
        transcription = result["text"]
        segments = result.get("segments", [])
        
    except Exception as e:
        logging.error("Error in audio transcription: %s", str(e))
        transcription = ""
        segments = []
        detected_language = "error"
    
    return transcription, segments, detected_language

def detect_audio_events(audio_file):
    try:
        waveform, sr = librosa.load(audio_file, sr=32000)
        segment_length = 5 * sr 
        events = {}
        for i in range(0, len(waveform), segment_length):
            segment = waveform[i:i + segment_length]
            if len(segment) == 0: continue
            segment_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0)
            output = panns_model.inference(segment_tensor)
            if isinstance(output, dict) and "clipwise_output" in output:
                clipwise_output = np.array(output["clipwise_output"], dtype=float)
            else:
                clipwise_output = np.array(output, dtype=float)
            if np.max(clipwise_output) < 0.1: continue
            top_idx = int(np.argmax(clipwise_output))
            event_label = pann_labels[top_idx] if top_idx < len(pann_labels) else "Unknown"
            timestamp = i / sr
            if event_label in events: events[event_label].append(seconds_to_timestr(timestamp))
            else: events[event_label] = [seconds_to_timestr(timestamp)]
        if not events: return {"No event": []}
        return events
    except Exception as e:
        logging.error("Error in audio event detection: %s", str(e))
        return {"Error": []}

def clean_report(text):
    text = re.sub(r'[\u06F0-\u06F9]+', '', text)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text

def describe_position(x_norm, y_norm):
    if x_norm < 0.33: horz = "left"
    elif x_norm < 0.66: horz = "center"
    else: horz = "right"
    if y_norm < 0.33: vert = "top"
    elif y_norm < 0.66: vert = "middle"
    else: vert = "bottom"
    return f"{horz}, {vert}"

def article_for(label):
    return "an" if label[0].lower() in "aeiou" else "a"

# --- Global Model Init ---
logging.info("Loading Whisper model (small)...")
whisper_model = whisper.load_model("small")
free_gpu()

logging.info("Loading PANNs audio detection model...")
panns_model = AudioTagging(checkpoint_path=PANN_MODEL_PATH)
free_gpu()
print_hardware_usage()

# --- UPDATED: Prompt to ask for quotes/timeline ---
def get_viral_prompt(report_text):
    return (
        f"""
        You are an expert Video Content Analyst.
        Analyze the provided video report (timestamped audio transcript & visual logs) to generate a comprehensive breakdown.

        --- CRITICAL TIMESTAMP LOGIC: MERGING SEGMENTS ---
        1. The transcript is split into small blocks (e.g., `[00:04:14 --> 00:04:24]`).
        2. A complete sentence often flows across multiple blocks.
        3. You MUST **MERGE** the timestamps for the full context.
        4. **METHOD**:
           - Identify the **First Block** where the quote starts. Take its **Start Time**.
           - Identify the **Last Block** where the quote ends. Take its **End Time**.
           - Combine them into one range: `[Start of First --> End of Last]`.
        
        *Example of Merging:*
        - Block A: `[00:04:14 --> 00:04:24] I think that video...`
        - Block B: `[00:04:24 --> 00:04:29] ...is the future.`
        - **YOUR OUTPUT**: `[00:04:14 --> 00:04:29]` (Start of A to End of B)
        -------------------------------------------------------------

        ### 1. VIDEO OVERVIEW
        * **Genre:** (Identify the genre)
        * **Summary:** (A cohesive narrative paragraph describing the flow)
        * **Target Audience:** (Who is this video for?)

        ### 2. ENGAGING DESCRIPTION (Social Media Ready)
        * **Hook:** (The opening line)
        * **Body:** (The core message/story)
        * **Key Takeaways:** (Bullet points of main features)

        ### 3. IMPORTANT TIMESTAMPS
        Select the most critical moments. 
        * **The Hook:** [Start --> End] - "Exact quote from transcript..." - (Description)
        * **The Conflict:** [Start --> End] - "Exact quote from transcript..." - (Description)
        * **The Climax/Solution:** [Start --> End] - "Exact quote from transcript..." - (Description)
        * **Key Highlight:** [Start --> End] - "Exact quote from transcript..." - (Description)
        * **The Viral Moment:** [Start --> End] - "Exact quote from transcript..." - (Description)
        * **Conclusion:** [Start --> End] - "Exact quote from transcript..." - (Description)

        ### 4. VIRAL EDIT STRATEGY
        * **Best Short Clip:** [Start Time --> End Time] (MERGE blocks if necessary to capture the full thought)
        * **Transcript:** (The full dialogue spoken during this merged timeframe)
        * **Reasoning:** (Why this segment is the best)

        ---
        REPORT DATA:
        {report_text}
        """
    )

# --- LLM Functions ---
def call_ollama(prompt, model):
    """Executes local Ollama model."""
    try:
        logging.info(f"Sending request to Ollama ({model})...")
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300 
        )
        if result.returncode != 0:
            logging.error("Ollama call failed: %s", result.stderr)
            return "LLM call failed."
        output = re.sub(r'<think>.*?</think>', '', result.stdout, flags=re.DOTALL).strip()
        return output if output else "No response received."
    except Exception as e:
        logging.error("Error calling Ollama: %s", str(e))
        return "Error in LLM call."

def call_openai(prompt, model="gpt-4o"):
    """Executes OpenAI model via API."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("OPENAI_API_KEY not found in environment variables.")
        return "Error: Missing OPENAI_API_KEY."

    logging.info(f"Sending request to OpenAI ({model})...")
    client = OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful video analysis assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"OpenAI API Error: {e}")
        return f"Error calling OpenAI: {e}"

def generate_video_descriptions(model_arg):
    """
    Decides which model backend to use based on the --model argument.
    """
    report_file = "report.txt"
    if not os.path.exists(report_file):
        logging.error("Report file not found.")
        return

    try:
        with open(report_file, "r", encoding="utf-8") as f:
            report_text = f.read()
    except Exception as e:
        logging.error(f"Error reading report: {e}")
        return

    clean_text = clean_report(report_text)
    shared_prompt = get_viral_prompt(clean_text)

    # --- SMART MODEL SELECTION LOGIC ---
    run_openai_flag = False
    run_ollama_flag = False
    
    # 1. Check for "Both"
    if model_arg.lower() == "both":
        run_openai_flag = True
        run_ollama_flag = True
        openai_model = "gpt-4o"
        ollama_model = "llama3" # Default if not specified
        logging.info("Model set to 'both'. Running OpenAI (gpt-4o) and Ollama (llama3).")

    # 2. Check for OpenAI (starts with gpt)
    elif model_arg.lower().startswith("gpt"):
        run_openai_flag = True
        openai_model = model_arg
        logging.info(f"Detected OpenAI model: {openai_model}")

    # 3. Default to Ollama for everything else (llama3, mistral, etc)
    else:
        run_ollama_flag = True
        ollama_model = model_arg
        logging.info(f"Detected Ollama model: {ollama_model}")

    # --- EXECUTION ---
    
    # Run Ollama
    if run_ollama_flag:
        ollama_summary = call_ollama(shared_prompt, model=ollama_model)
        try:
            with open("video_description_ollama.txt", "w", encoding="utf-8") as f:
                f.write(f"--- Generated by Ollama ({ollama_model}) ---\n\n")
                f.write(ollama_summary)
            logging.info("Saved video_description_ollama.txt")
        except Exception as e:
            logging.error(f"Error saving Ollama description: {e}")

    # Run OpenAI
    if run_openai_flag:
        openai_summary = call_openai(shared_prompt, model=openai_model)
        try:
            with open("video_description_openai.txt", "w", encoding="utf-8") as f:
                f.write(f"--- Generated by OpenAI ({openai_model}) ---\n\n")
                f.write(openai_summary)
            logging.info("Saved video_description_openai.txt")
        except Exception as e:
            logging.error(f"Error saving OpenAI description: {e}")

# --- UPDATED: Process Video (Deduplication + Timeline Write) ---
def process_video(video_path, model_name, sample_rate=1):
    if not os.path.exists(video_path):
        logging.error("File does not exist: %s", video_path)
        return

    # --- FIX: Convert Corrupt/AV1 Video to H.264 automatically ---
    import imageio_ffmpeg 
    fixed_video_path = os.path.splitext(video_path)[0] + "_fixed_h264.mp4"
    
    if not os.path.exists(fixed_video_path):
        logging.info(f"Checking/Converting video format for stability: {video_path}")
        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            subprocess.run([
                ffmpeg_exe, "-y", "-i", video_path,
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                "-c:a", "aac", fixed_video_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            logging.info("Video conversion complete.")
        except Exception as e:
            logging.error(f"Video conversion failed: {e}")
            logging.warning("Proceeding with original file.")
            fixed_video_path = video_path 

    video_path = fixed_video_path

    # --- Video Properties ---
    clip = VideoFileClip(video_path)
    fps = clip.fps
    frame_count = int(clip.reader.n_frames)
    duration = clip.duration
    width, height = clip.size
    video_format = os.path.splitext(video_path)[1].lower()
    clip.close()

    logging.info("Video properties: Duration: %s, Frames: %d, FPS: %.2f", seconds_to_timestr(duration), frame_count, fps)

    # --- Report Setup ---
    report_lines = []
    report_lines.append("Video Processing Report")
    report_lines.append(f"File: {video_path}")
    report_lines.append(f"Duration: {seconds_to_timestr(duration)}")
    report_lines.append("")

    # --- Audio Analysis ---
    temp_audio_path = "temp_audio.wav"
    logging.info("Extracting audio...")
    extract_audio(video_path, temp_audio_path)

    audio_transcript = ""
    audio_segments = [] # Variable for segments
    
    if os.path.exists(temp_audio_path):
        logging.info("Transcribing audio...")
        # Unpack the 3 values
        audio_transcript, audio_segments, _ = transcribe_audio(temp_audio_path)
        
        logging.info("Detecting audio events...")
        audio_events = detect_audio_events(temp_audio_path)
        os.remove(temp_audio_path)
    else:
        audio_events = {"No audio": []}

    # --- NEW: Write detailed transcript to report ---
    report_lines.append("=== TIMELINED AUDIO TRANSCRIPT ===")
    if audio_segments:
        for seg in audio_segments:
            start_str = seconds_to_timestr(seg['start'])
            end_str = seconds_to_timestr(seg['end'])
            text = seg['text'].strip()
            report_lines.append(f"[{start_str} --> {end_str}] {text}")
    else:
        report_lines.append("No dialogue detected.")
    
    report_lines.append("\n=== AUDIO EVENTS ===")
    report_lines.append(f"{audio_events}")
    report_lines.append("\n=== VISUAL LOGS ===")
    
    free_gpu()

    # --- Visual Models ---
    logging.info("Loading Visual Models...")
    yolo_engine = get_yolo_model()
    blip_gen, blip_proc = get_blip_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Process Video Frames ---
    cap = cv2.VideoCapture(video_path)
    
    frame_idx = 0
    prev_hist = None
    last_logged_time = -10
    
    # --- FIX: Initialize variable to track the last text ---
    last_caption = None 
    
    SIMILARITY_THRESHOLD = 0.90 
    FORCE_LOG_INTERVAL = 5.0 

    logging.info("Starting visual analysis...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        if frame_idx % sample_rate != 0:
            continue

        current_time = frame_idx / fps
        
        # Scene Detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

        is_duplicate = False
        if prev_hist is not None:
            similarity = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if similarity > SIMILARITY_THRESHOLD:
                if (current_time - last_logged_time) < FORCE_LOG_INTERVAL:
                    is_duplicate = True
        
        if is_duplicate:
            continue

        prev_hist = hist
        last_logged_time = current_time
        time_str = seconds_to_timestr(current_time)

        # 1. YOLO
        results = yolo_engine(frame, verbose=False)
        yolo_descriptions = []
        for result in results:
            if result.boxes and result.boxes.data is not None:
                for det in result.boxes.data.cpu().numpy():
                    label = CLASS_MAP.get(int(det[5]), "Unknown")
                    yolo_descriptions.append(label)
        
        yolo_text = ", ".join(list(set(yolo_descriptions))) if yolo_descriptions else None

        # 2. BLIP
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        inputs = blip_proc(pil_img, return_tensors="pt").to(device)
        try:
            output_ids = blip_gen.generate(**inputs, max_length=60, min_length=15, num_beams=5)
            caption = blip_proc.decode(output_ids[0], skip_special_tokens=True)
            
            # --- FIX: Check for duplicate caption ---
            if caption and caption == last_caption:
                caption = None # Suppress this specific caption from the log
            elif caption:
                last_caption = caption # Update last seen caption
                
        except:
            caption = None

        # 3. Log
        log_parts = []
        if yolo_text: log_parts.append(f"Objects: [{yolo_text}]")
        if caption: log_parts.append(f"Scene: {caption}")
        
        if log_parts:
            log_line = f"Time {time_str}: " + " | ".join(log_parts)
            logging.info(log_line)
            report_lines.append(log_line)

    cap.release()
    logging.info("Finished visual processing.")
    free_gpu()

    # --- Write Final Report ---
    report_filename = "report.txt"
    try:
        with open(report_filename, "w", encoding="utf-8") as rpt:
            rpt.write("\n".join(report_lines))
        logging.info("Report generated.")
    except Exception as e:
        logging.error(f"Error writing report: {e}")

    # --- Generate Summaries (Smart Selection) ---
    generate_video_descriptions(model_name)
    free_gpu()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Processing AI Tool (CLI Only)")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file to process")
    
    # Updated help text to reflect new "Smart Model" selection
    parser.add_argument(
        "--model", 
        type=str, 
        default="llama3", 
        help="Model to use. Examples: 'llama3' (Ollama), 'gpt-4o' (OpenAI), or 'both' (Runs both)."
    )
    
    parser.add_argument("--sample_rate", type=int, default=32, help="Process every Nth frame (default: 32)")
    
    args = parser.parse_args()

    print(f"--- Running Video Processing ---")
    print(f"Input: {args.video}")
    print(f"Model Selection: {args.model}")
    print(f"Sample Rate: {args.sample_rate}")
    
    try:
        process_video(
            video_path=args.video,
            model_name=args.model,
            sample_rate=args.sample_rate
        )
        print("\n--- Processing Complete ---")
        print("Results saved.")
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")