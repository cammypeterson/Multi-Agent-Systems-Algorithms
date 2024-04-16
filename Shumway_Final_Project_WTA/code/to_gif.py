import os
from PIL import Image

def png_to_gif(folder_path, output_path):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)
    
    # Filter only the PNG files
    png_files = sorted([f for f in files if f.lower().endswith('.png')])
    
    if not png_files:
        print("No PNG files found in the directory.")
        return
    
    images = []
    for png_file in png_files:
        # Open each PNG file
        image = Image.open(os.path.join(folder_path, png_file))
        # Convert to RGB if not already in that mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        images.append(image)
    
    # Save as GIF
    if images:
        images[0].save(output_path, save_all=True, append_images=images[1:], duration=25, loop=0)
        print(f"GIF saved successfully at {output_path}")
    else:
        print("No images found to create GIF.")


folder_path = "frames/completion/" # folder where frames are stored as PNG files
output_path = "completion.gif" # where to output GIF file
png_to_gif(folder_path, output_path)