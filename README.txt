This is a simple facial landmark finder which uses PyTorch and Resnet18 to find three landmarks. It works surprisingly well.

--------------------------------------------------------------------------

To use it:

import Landmark3

# initialise
L = Landmark3.Landmark3(weightfile)

# find 
lx,ly,rx,ry,nx,ny = L.process(image,rect)

--------------------------------------------------------------------------

The pretrained weights file is for finding left and right eye centres and nose tips.
(N.B. "left" and "right" are relative to the subject, not to the image - the "left eye" is the one on the right of the image.)

--------------------------------------------------------------------------

train3landmarks.py trains the Landmark3 model

call:
python train3landmarks.py Landmark3 lists/landmark_train.txt

parameters:
1: model name, used in saving weights files (e.g. "Landmark3")
2: name of a text file containing a list of training data (e.g. "lists/landmark_train.txt")

Each line of the training data file contains:
1: filename of an image file
2: x-coordinate of landmark 1 in the image
3: y-coordinate of landmark 1 in the image
4: x-coordinate of landmark 2 in the image
5: y-coordinate of landmark 2 in the image
6: x-coordinate of landmark 3 in the image
7: y-coordinate of landmark 3 in the image

--------------------------------------------------------------------------

test3landmarks.py is a demo program to show the landmarks found on a directory of images

call:
python test3landmarks.py pretrained/Landmarks3.pt TestInput/ 20

parameters:
1: name of the weights file (e.g. "pretrained/Landmarks3.pt")
2: name of the directory of images (e.g. "TestInput/")
3: delay between images, in msec (e.g. "20")

The demo uses the Yolo v3 face finder to locate faces in images.

--------------------------------------------------------------------------
