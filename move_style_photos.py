import os
from PIL import Image

generated_photos = os.listdir('/home/davidg/style_transfer_project/neural-style-tf/image_output')

# photo name
# generated image path
# convert to jpg
# save to new folder

input_dir_path = '/home/davidg/style_transfer_project/neural-style-tf/image_output/'
output_dir_path ='/home/davidg/style_transfer_project/generated_photos_round_1/'

for i, photo in enumerate(generated_photos):
    img_path = input_dir_path + photo + '/' + photo + '.png'
    with Image.open(img_path) as png_image:
        png_image.save(output_dir_path + photo, "JPEG")
    print( i ) 
