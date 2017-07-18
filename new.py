import os
from PIL import Image


a = open("Volkswagen.txt", "w")
i=0
for path, subdirs, files in os.walk(r'/home/praphulr/car_logo/Volkswagen'):
   for filename in files:
      #img = Image.open(filename)
      with Image.open(os.path.join(path, filename)) as img:
      #width, height = img.size
       k = img.size
 
       
      a.write(filename + ' ' + 'Volkswagen' +' '+ str(int(i/10)+1) + ' ' + '0'  + ' ' + '0' + ' ' + str(k[1]) + ' ' + str(k[0]) + os.linesep) 
      i=i+1


