from moviepy import VideoFileClip, concatenate_videoclips
import os

def create_highlight_reel(input_filename, output_filename):
    # Check if file exists
    if not os.path.exists(input_filename):
        print(f"Error: Input file '{input_filename}' not found.")
        return

    try:
        # Load the source video
        with VideoFileClip(input_filename) as video:
            
            # Define the cuts based on the "Trailer" narrative structure
            # Format: (Start Time, End Time) in "HH:MM:SS"
            cuts_timeframes = [
                ("00:00:30", "00:01:08"), # 1. The Hook: Wannabe Entrepreneurs & Context
                ("00:05:05", "00:06:14"), # 2. The Conflict: Twitter's Direction & Bias
                ("00:08:00", "00:09:03"), # 3. The Climax: Creating Collective Consciousness
                ("00:13:38", "00:14:17"), # 4. Key Highlight: Humans as Cellular Organization
                ("00:20:20", "00:20:27"), # 5. The Viral Moment: Flashlight Analogy (Intro)
                ("00:20:21", "00:21:02"), # 6. Best Short Clip: Full Flashlight/Satellite Segment
                ("00:28:45", "00:29:58"), # 7. Conclusion: AI, Urbanization & Future Outlook
            ]
            
            '''
            cuts_timeframes = [
                # 1. The Simulation (The "Video Games" Argument)
                # "If you look at the advancement of video games... indistinguishable from reality."
                ("00:29:30", "00:30:45"), 

                # 2. The End of Money (Energy is Currency)
                # "Money disappears as a concept... Energy is the true currency."
                ("00:12:45", "00:13:45"), 

                # 3. True Friendship (The "Chips Down" Quote)
                # "Everyone likes you when the chips are up, but who likes you when the chips are down?"
                ("01:18:45", "01:19:30"), 

                # 4. Government Waste (DOGE & $100 Billion Savings)
                # "Department of Government Efficiency... making payments audit-able."
                ("01:31:00", "01:32:00"), 

                # 5. Advice to Builders (Value Creation)
                # "Aim to make more than you take."
                ("01:46:30", "01:47:30"),
            ]
            
            cuts_timeframes = [
                ("00:00:01", "00:04:30"), # 1. The Hook: Intro & Tone (4m 29s)
                ("00:05:05", "00:09:35"), # 2. The Conflict: Social Media Bias (4m 30s)
                ("00:12:10", "00:16:40"), # 3. The Solution: Future of Money & Energy (4m 30s)
                ("01:00:27", "01:05:00"), # 4. Key Feature: Simulation Theory (4m 33s)
                ("01:10:26", "01:15:00"), # 5. Viral Moment: Humor & AI (4m 34s)
                ("01:54:02", "01:54:13"), # 6. Ending: Closure (Uses remaining file duration)
            ]
            cuts_timeframes = [
                ("01:00:27", "01:01:25"), # 1. The Mind-Bender: "Video games indistinguishable from reality" (58s)
                ("00:12:30", "00:13:20"), # 2. The Future: "Money disappears... Energy is the currency" (50s)
                ("01:16:10", "01:16:55"), # 3. Real Talk: "Who likes you when the chips are down?" (45s)
                ("01:34:10", "01:35:00"), # 4. Controversy: DOGE & Government Waste (50s)
                ("01:49:45", "01:50:40"), # 5. Motivation: "Make more than you take" (55s)
            ]
            
            cuts_timeframes = [
                ("00:29:30", "00:34:30"), # 1. The Hook: Simulation Theory
                ("00:22:00", "00:28:00"), # 2. The Vision: Economy & AI
                ("00:12:30", "00:15:30"), # 3. The Identity: "X" & Energy
                ("01:34:00", "01:36:30"), # 4. The Struggle: DOGE & Bureaucracy
                ("01:16:00", "01:19:00"), # 5. The Personal: Friendship
                ("01:49:30", "01:52:30"), # 6. The Closer: Advice to Builders
            ]
            '''
            print(f"Processing {len(cuts_timeframes)} clips from {input_filename}...")
            
            clips = []
            for i, (start, end) in enumerate(cuts_timeframes):
                print(f"  - Cutting Clip {i+1}: {start} to {end}")
                
                # --- FIX IS HERE ---
                # In MoviePy v2.0+, use .subclipped() instead of .subclip()
                clip = video.subclipped(start, end)
                clips.append(clip)

            print("Stitching clips together...")
            final_video = concatenate_videoclips(clips)

            print(f"Writing output to {output_filename} (this may take a few minutes)...")
            final_video.write_videofile(
                output_filename, 
                codec="libx264", 
                audio_codec="aac",
                fps=24,
                preset="medium",
                threads=4
            )
            
            print("Done! Video saved successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")
        # Print full traceback if needed for further debugging
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    input_file = "musk.mp4" 
    output_file = "musk_5.mp4"
    
    create_highlight_reel(input_file, output_file)