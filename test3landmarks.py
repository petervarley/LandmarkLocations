import cv2
import os
import numpy as np
import math
import sys
import Yolo as Face
import Landmark3

############################################################################################

def box_size_from_iod (iod):
	ew = iod * 0.6
	eh = iod * 0.6
	return (ew,eh)

############################################################################################

def draw_face (image,ix,iy,iw,ih):
	cv2.rectangle(image,(int(ix),int(iy)),(int(ix+iw),int(iy+ih)),(0,255,0),2)

############################################################################################

def draw_3_landmarks (image,lx,ly,rx,ry,nx,ny,w,h):
	cv2.rectangle(image,(int(lx-w/2),int(ly-h/2)),(int(lx+w/2),int(ly+h/2)),(0,255,255),3)
	cv2.rectangle(image,(int(rx-w/2),int(ry-h/2)),(int(rx+w/2),int(ry+h/2)),(0,255,255),3)
	cv2.rectangle(image,(int(nx-w/2),int(ny-h/2)),(int(nx+w/2),int(ny+h/2)),(0,255,255),3)

############################################################################################

def test_landmark_on_directory(weightfile,directory_name,wait=100):
	L = Landmark3.Landmark3(weightfile)
	face_finder = Face.startup()

	for f in os.listdir(directory_name):
		if f.endswith('.jpg') or f.endswith('.png'):
			filename=os.path.join(directory_name,f)
			rgbframe = cv2.imread(filename)
			faces_found = Face.process(face_finder,rgbframe)

			for (ix,iy,iw,ih) in faces_found:
				if (ix >= 0) and (iy >= 0):
					lx,ly,rx,ry,nx,ny = L.process(rgbframe,(ix,iy,iw,ih))
					lw,lh = box_size_from_iod(lx-rx)

					draw_face(rgbframe,ix,iy,iw,ih)
					draw_3_landmarks(rgbframe,lx,ly,rx,ry,nx,ny,lw,lh)

			cv2.imshow('Face',rgbframe)
			cv2.moveWindow('Face',200,200)

			c = cv2.waitKey(wait)
			if c == 27:
				break

############################################################################################

def test_best_landmark_on_directory(weightfile,directory_name,wait=100):
	L = Landmark3.Landmark3(weightfile)
	face_finder = Face.startup()

	for f in os.listdir(directory_name):
		if f.endswith('.jpg') or f.endswith('.JPG') or f.endswith('.png'):
			filename=os.path.join(directory_name,f)
			rgbframe = cv2.imread(filename)
			ret, (ix,iy,iw,ih) = Face.best_face(face_finder,rgbframe)

			if ret and (ix >= 0) and (iy >= 0):
				lx,ly,rx,ry,nx,ny = L.process(rgbframe,(ix,iy,iw,ih))
				lw,lh = box_size_from_iod(lx-rx)

				draw_face(rgbframe,ix,iy,iw,ih)
				draw_3_landmarks(rgbframe,lx,ly,rx,ry,nx,ny,lw,lh)

			cv2.imshow('Face',rgbframe)
			cv2.moveWindow('Face',200,200)

			c = cv2.waitKey(wait)
			if c == 27:
				break

############################################################################################

def test_central_landmark_on_directory(weightfile,directory_name,wait=100):
	L = Landmark3.Landmark3(weightfile)

	face_finder = Face.startup()

	for f in os.listdir(directory_name):
		if f.endswith('.jpg') or f.endswith('.png'):
			filename=os.path.join(directory_name,f)
			rgbframe = cv2.imread(filename)
			centre = (int(rgbframe.shape[1]/2),int(rgbframe.shape[0]/2))
			ret, (ix,iy,iw,ih) = Face.face_around(face_finder,rgbframe,centre)

			if ret and (ix >= 0) and (iy >= 0):
				lx,ly,rx,ry,nx,ny = L.process(rgbframe,(ix,iy,iw,ih))
				lw,lh = box_size_from_iod(lx-rx)

				draw_face(rgbframe,ix,iy,iw,ih)
				draw_3_landmarks(rgbframe,lx,ly,rx,ry,nx,ny,lw,lh)

			cv2.imshow('Face',rgbframe)
			cv2.moveWindow('Face',200,200)

			c = cv2.waitKey(wait)
			if c == 27:
				break

############################################################################################

if __name__ == '__main__':
	if os.path.isdir(sys.argv[2]):
		test_landmark_on_directory(sys.argv[1],sys.argv[2],int(sys.argv[3]))
		#test_best_landmark_on_directory(sys.argv[1],sys.argv[2],int(sys.argv[3]))
		#test_central_landmark_on_directory(sys.argv[1],sys.argv[2],int(sys.argv[3]))
	else:
		pass

############################################################################################
# Usage: python test3landmarks.py pretrained/Landmarks3.pt TestInput/ 20
