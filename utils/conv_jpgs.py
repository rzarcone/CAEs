from PIL import Image
import fnmatch
import os
import re
import numpy as np

flickr_img_folder = "/media/tbell/datasets/flickr_yfcc100m/images/"

for root, dirnames, filenames in os.walk(flickr_img_folder):
  for filename in fnmatch.filter(filenames, "*.jpg"):
    input_file = os.path.join(root, filename)
    new_root = re.sub("images", "proc_images", root)
    if not os.path.exists(new_root):
      os.makedirs(new_root)
    new_output_loc = os.path.join(new_root, filename)
    if not os.path.exists(new_output_loc):
      try:
        img = Image.open(input_file)
      except:
        #print("Could not load"+input_file)
        continue
      arry = np.array(img, dtype=np.uint8)
      new_img = Image.fromarray(arry)
      new_img.save(new_output_loc)
    print(new_output_loc)
