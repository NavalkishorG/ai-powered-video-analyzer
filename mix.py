import os
import json
import numpy as np
from moviepy import VideoFileClip, ImageClip, ColorClip, concatenate_videoclips
from PIL import Image, ImageDraw, ImageFont

def create_text_image(text, width, height, duration=3):
    """Generates a text card using PIL (White text on Black background)."""
    img = Image.new('RGB', (width, height), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Load Font
    try:
        font_size = int(height / 10)
        # Try loading Arial, fallback to default if missing
        font = ImageFont.truetype("arial.ttf", size=font_size)
    except IOError:
        font = ImageFont.load_default()

    # Calculate Text Position (Center)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (width - text_w) / 2
    y = (height - text_h) / 2
    
    draw.text((x, y), text, font=font, fill=(255, 255, 255))
    
    return ImageClip(np.array(img)).with_duration(duration)

def create_highlight_reel(input_filename, output_filename, cuts_json="viral_cuts.json"):
    # 1. Validation
    if not os.path.exists(input_filename):
        print(f"Error: Input file '{input_filename}' not found.")
        return
    if not os.path.exists(cuts_json):
        print(f"Error: '{cuts_json}' not found. Run video_processing_gui.py first.")
        return

    # 2. Load Cuts
    with open(cuts_json, "r") as f:
        cuts_data = json.load(f)

    print(f"Loaded {len(cuts_data)} clips from JSON.")

    try:
        with VideoFileClip(input_filename) as video:
            w, h = video.size
            final_sequence = []

            for i, clip_data in enumerate(cuts_data):
                start = clip_data['start']
                end = clip_data['end']
                text_overlay = clip_data['description']
                
                print(f"Processing Clip {i+1}: {text_overlay} ({start} -> {end})")

                # --- A. Description Frame (3 Seconds) ---
                # "What is to be coming in next frame"
                desc_clip = create_text_image(text_overlay, w, h, duration=3)
                final_sequence.append(desc_clip)

                # --- B. The Video Clip ---
                video_clip = video.subclipped(start, end)
                final_sequence.append(video_clip)

                # --- C. Black Screen (1 Second) ---
                black_screen = ColorClip(size=(w, h), color=(0,0,0)).with_duration(1)
                final_sequence.append(black_screen)

            # 3. Concatenate and Write
            print("Stitching video...")
            final_video = concatenate_videoclips(final_sequence, method="compose")
            
            print(f"Saving to {output_filename}...")
            final_video.write_videofile(
                output_filename, 
                codec="libx264", 
                audio_codec="aac",
                fps=24,
                preset="medium",
                threads=4
            )
            print("Done!")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Ensure this matches your downloaded video file
    input_video = "musk30.mp4" 
    output_video = "musk_edit.mp4"
    
    
    create_highlight_reel(input_video, output_video)
    