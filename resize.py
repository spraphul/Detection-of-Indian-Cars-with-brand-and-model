# Copyright 2017 The Impetuors Authors. All Rights Reserved.
# Authors:
# Preetham Paul Sunkari, Praphul Singh, Harshit Saini, Aadrish Sharma
# Limited under the License.
##########################################################################################
import PIL
from PIL import Image
import os, sys

def resizeImage(infile, output_dir="", size=(200,200)):
     outfile = os.path.splitext(infile)[0]+"_resized"
     extension = os.path.splitext(infile)[1]

     #if (cmp(extension, ".jpg")):
      #  return

     if infile != outfile:
        try :
            im = Image.open(infile)
            im.resize(size, Image.ANTIALIAS)
            im.save(output_dir+outfile+extension,"jpg")
        except IOError:
            print("cannot reduce image for ", infile)


if __name__=="__main__":
    output_dir = "Paul"
    dir = "/home/praphulr/DeepLogo-master/Images/Images"

    if not os.path.exists(os.path.join(dir,output_dir)):
        os.mkdir(output_dir)

    for file in os.listdir(dir):
        resizeImage(file,output_dir)
