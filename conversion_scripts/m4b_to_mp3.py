from pydub import AudioSegment

def convert_m4b_to_mp3(input_path, output_path):
    # Load the .m4b file
    audio = AudioSegment.from_file(input_path, format="m4a")
    
    # Export as .mp3
    audio.export(output_path, format="mp3")
    print(f"Converted {input_path} to {output_path}")

if __name__ == "__main__":
    input_file = "The Worlds I See_ Curiosity, Exploration, and Discovery at the Dawn of AI by Fei-Fei Li M4B\\The Worlds I See_ Curiosity, Exploration, and Discovery at the Dawn of AI by Fei-Fei Li.m4a"
    output_file = "The Worlds I See_ Curiosity, Exploration, and Discovery at the Dawn of AI by Fei-Fei Li M4B\\The Worlds I See.mp3"
    convert_m4b_to_mp3(input_file, output_file)
