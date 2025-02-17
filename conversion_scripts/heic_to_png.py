import os
from PIL import Image
import pillow_heif  # This enables HEIC/HEIF support in Pillow

def convert_heic_to_png(input_folder):
    """
    Convert all .heic images in the input_folder to .png.
    """
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".heic"):
            heic_path = os.path.join(input_folder, filename)
            
            # Open the HEIC file via pillow_heif
            heif_file = pillow_heif.open_heif(heic_path)
            
            # Convert HEIC data to a Pillow Image
            image = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                "raw",
                heif_file.mode,
                heif_file.stride,
            )

            # Replace the .heic extension with .png
            png_filename = os.path.splitext(filename)[0] + ".png"
            png_path = os.path.join(input_folder, png_filename)

            # Save as PNG
            image.save(png_path, format="PNG")
            print(f"Converted {filename} -> {png_filename}")

if __name__ == "__main__":
    folder_to_convert = "C:\\Users\\Anthony\\Downloads\\NZ Trip-20241222T094110Z-001\\NZ Trip"  # You can specify another folder if needed
    convert_heic_to_png(folder_to_convert)
