# %%
# Imports
from PIL import Image
import os

def resize_image(final_size, im):
    '''
    Resizes a given image to a final size.
    '''

    size = im.size
    ratio = float(final_size) / max(size)
    new_image_size = tuple([int(x*ratio) for x in size])
    im = im.resize(new_image_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (final_size, final_size))
    new_im.paste(im, ((final_size-new_image_size[0])//2, (final_size-new_image_size[1])//2))
    return new_im

# %%
# Only run this code if running whole script
if __name__ == '__main__':

    path = "data/images/"
    cleaned_path = "data/cleaned_images/"
    bw_path = "data/bw_images/"

    # Resize images to 512px (normalisation)
    dirs = os.listdir(path)
    cleaned_dirs = os.listdir(cleaned_path)
    final_size = 512

    # for n, item in enumerate(dirs, 1):
    #     # If the cleaned image does not exist, clean it
    #     cleaned_img_path = cleaned_path + item
    #     if not os.path.exists(cleaned_img_path):
    #         im = Image.open(path + item)
    #         new_im = resize_image(final_size, im)

    #         # Turn to B&W (1 channel only)
    #         new_im.convert("L")
    #         new_im.save(cleaned_img_path)
    #         print(str(n) + ' images converted')

    for n, item in enumerate(cleaned_dirs, 1):
        # If a B&W image does not exist, make one
        bw_img_path = bw_path + item
        if not os.path.exists(bw_img_path):
            im = Image.open(cleaned_path + item)
            im = im.convert("L")
            im.save(bw_img_path)
            print(str(item) + ' converted to B&W')
            print(str(n) + ' images converted')
# %%
