# Copyright 2017 The Impetuors Authors. All Rights Reserved.
# Authors:
# Preetham Paul Sunkari, Praphul Singh, Harshit Saini, Aadrish Sharma
# Limited under the License.
##########################################################################################

import os
path = '/home/praphulr/Chevrolet'
files = os.listdir(path)
i = 1

for file in files:
    os.rename(os.path.join(path, file), os.path.join(path, str(i)+'.jpg'))
    i = i+1
