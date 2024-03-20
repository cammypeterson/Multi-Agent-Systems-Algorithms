import os
from PIL import Image

# Class used to generate GIF from image files saved in the images/ directory
class GIFGenerator:
    def __init__(self):
        desired_dir = "Connectedness of Agents"
        if os.path.split(os.getcwd())[1] != desired_dir: # if you are not in the desired directory
            if os.path.isdir(desired_dir): # if the desired directory is a subdirectory of the current directory 
                os.chdir(desired_dir) # change to the desired directory
            elif desired_dir in os.getcwd(): # if the desired directory is above you
                
                # move up directories until you are at the desired directory
                while os.path.split(os.getcwd())[1] != desired_dir: 
                    os.chdir(os.path.split(os.getcwd())[0])
        
                  
        self.directory_path = os.path.join(os.getcwd(), "images") 
        if not os.path.isdir(self.directory_path): # create the images/ directory
            os.mkdir(self.directory_path) # go to the images/ directory
    
    # Generate GIF from the image files saved in the images/ directory
    def create_gif(self, gif_name):
        # List all the PNG files in the specified folder
        image_list = sorted([f"{self.directory_path}/{filename}" for filename in os.listdir(self.directory_path) if filename.endswith('.png')])
        
        frames = []
        for image_name in image_list:
            frames.append(Image.open(image_name))

        # create GIF/ directory if there isn't one
        if not os.path.isdir('GIF'):
            os.mkdir('GIF')
        frames[0].save(gif_name, format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=0)

    # delete the .png files in the images/ directory
    def delete_frames(self):
        # Change working directory to "Connectedness of Agents" directory, or stay where you are if it does not exist
        
        # delete the files in the image directory
        files = os.listdir(self.directory_path) 
        for file in files:
                file_path = os.path.join(self.directory_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        