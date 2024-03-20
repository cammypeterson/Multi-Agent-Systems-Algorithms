import os
from PIL import Image

# Class used to generate GIF from image files saved in the images/ directory
class GIFGenerator:
    def __init__(self):
        self.directory_path = os.path.join(os.getcwd(), "images")
    
    # Generate GIF from the image files saved in the images/ directory
    def create_gif(self, gif_name):
        # List all the PNG files in the specified folder
        image_list = sorted([f"{self.directory_path}/{filename}" for filename in os.listdir(self.directory_path) if filename.endswith('.png')])
        
        frames = []
        for image_name in image_list:
            frames.append(Image.open(image_name))

        frames[0].save(gif_name, format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=0)

    # delete the .png files in the images/ directory
    def delete_frames(self):
        directory_path = os.path.join(os.getcwd(), "images")
        if not os.path.isdir(directory_path):
            os.mkdir(directory_path)
        files = os.listdir(directory_path)
        for file in files:
                file_path = os.path.join(directory_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        