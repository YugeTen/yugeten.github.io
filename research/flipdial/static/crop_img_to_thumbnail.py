from PIL import Image
import os

imgdir = os.path.join('images')
thumbdir = os.path.join('thumbnails')
imglist = os.listdir(imgdir)

down_size = 400,400
crop_height = 200
crop_width = 200

for img_i in imglist: 
    try:
        img = Image.open(os.path.join(imgdir, img_i))
        img.thumbnail(down_size)
        width, height = img.size

        left = (width - crop_width)/2
        top = (height - crop_height)/2
        right = (width + crop_width)/2
        bottom = (height + crop_height)/2

        img_cropped = img.crop((left, top, right, bottom))
        img_cropped.save(os.path.join(thumbdir, img_i), "JPEG")

    except IOError:
        print "cannot create thumbnail for '%s'" % img_i

