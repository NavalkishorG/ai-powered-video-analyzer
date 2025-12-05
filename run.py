import subprocess
import sys
import os
import time

def run_pipeline(video_filename, model_type="gpt-4o"):
    """
    Executes the Video AI Pipeline:
    1. Analyzes video & generates cuts (video_processing_gui.py)
    2. Edits the video based on those cuts (mix.py)
    """
    
    print(f"=================================================")
    print(f"   🚀 STARTING AI VIDEO PIPELINE: {video_filename}")
    print(f"=================================================\n")

    # --- STEP 1: ANALYSIS & CUT GENERATION ---
    print(f"--- [Step 1/2] Analyzing Video with {model_type} ---")
    print("    * This may take a few minutes depending on video length...")
    
    analyze_cmd = [
        sys.executable, "video_processing_gui.py",
        "--video", video_filename,
        "--model", model_type,
        "--sample_rate", "250"  # Process 1 frame every 10s for speed
    ]

    try:
        # check=True will stop the pipeline if this script fails
        subprocess.run(analyze_cmd, check=True) 
        print("    ✅ Analysis complete. 'viral_cuts.json' and 'viral_report.txt' created.\n")
        
    except subprocess.CalledProcessError as e:
        print(f"\n    ❌ CRITICAL ERROR in Analysis Step. Pipeline stopped.")
        print(f"    Error details: {e}")
        return

    # --- Verification Check ---
    if not os.path.exists("viral_cuts.json"):
        print("    ❌ Error: 'viral_cuts.json' was not found. The analysis script failed to save it.")
        return

    # --- STEP 2: VIDEO EDITING ---
    print(f"--- [Step 2/2] Editing & Stitching Video ---")
    print("    * Reading cut list and rendering final output...")

    # We are calling mix.py. 
    # Note: Ensure mix.py is set to read 'viral_cuts.json' by default as per previous code.
    edit_cmd = [sys.executable, "mix.py"]

    try:
        subprocess.run(edit_cmd, check=True)
        print("    ✅ Editing complete.\n")
        
    except subprocess.CalledProcessError as e:
        print(f"\n    ❌ CRITICAL ERROR in Editing Step.")
        print(f"    Error details: {e}")
        return

    print("=================================================")
    print(f"   🎉 PIPELINE FINISHED!")
    print(f"   Check your folder for the final output video.")
    print("=================================================")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Change this to your actual video file name
    TARGET_VIDEO = "musk30.mp4" 
    
    # Run the pipeline
    run_pipeline(TARGET_VIDEO, model_type="gpt-4o")