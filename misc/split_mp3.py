from mutagen.mp3 import MP3
from pydub import AudioSegment
import ffmpeg
import os

# Your array of MP3 file paths
inp_folder = "C:\\Users\\Anthony\\Downloads\\Brandon Sanderson - Mistborn 02 - The Well of Ascension [Graphic Audio]"
sav_folder = os.path.join(inp_folder, "split_mp3")
os.makedirs(sav_folder, exist_ok=True)


mp3_files = os.listdir(inp_folder)

def get_duration(file_path):
    # Get duration in seconds using ffmpeg
    probe = ffmpeg.probe(file_path)
    return float(probe['format']['duration'])

def split_mp3(file_path):
    base_name = os.path.splitext(file_path)[0]  # Remove .mp3 extension
    mp3 = MP3(os.path.join(inp_folder, file_path))

    # Check for chapter metadata
    chapters = mp3.get('CHAP', None)
    if chapters:
        print(f"Found chapters in {file_path}")
        for i, chapter in enumerate(chapters, 1):
            start_time = chapter.start_time / 1000  # Convert to seconds
            end_time = chapter.end_time / 1000
            output_file = f"{base_name}_{i:03d}.mp3"
            output_file = os.path.join(sav_folder, output_file)
            try:
                stream = ffmpeg.input(os.path.join(inp_folder, file_path), ss=start_time, t=end_time - start_time)
                stream = ffmpeg.output(stream, output_file, c='copy', loglevel='quiet')
                ffmpeg.run(stream)
                print(f"Saved: {output_file}")
            except ffmpeg.Error as e:
                print(f"Error splitting {file_path}: {e}")
    else:
        # No chapters, split into 15-minute segments
        print(f"No chapters found in {file_path}, splitting into 15-minute segments")
        segment_length = 15 * 60  # 15 minutes in seconds
        total_duration = get_duration(os.path.join(inp_folder, file_path))
        
        for i in range(0, int(total_duration), segment_length):
            start = i
            if start == 0:
                start = 1
            #else:
            #    continue
            duration = min(segment_length, total_duration - start)
            output_file = f"{base_name}_{(i // segment_length + 1):03d}.mp3"
            output_file = os.path.join(sav_folder, output_file)
            try:
                stream = ffmpeg.input(os.path.join(inp_folder, file_path), ss=start, t=duration)
                stream = ffmpeg.output(stream, output_file, c='copy', loglevel='quiet')
                ffmpeg.run(stream)
                print(f"Saved: {output_file}")
            except ffmpeg.Error as e:
                print(f"Error splitting {file_path}: {e}")

# Process each file
for file in mp3_files:
    split_mp3(file)