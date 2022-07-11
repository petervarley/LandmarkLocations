import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import math
from resnet import resnet18

############################################################################################

class Coordinates(nn.Module):
    def __init__(self):
        super(Coordinates, self).__init__()
        self.number_of_outputs = 6

        self.img_feature_dim = 256  # the dimension of the CNN feature to represent each frame

        self.base_model = resnet18(pretrained=True)

        self.base_model.fc2 = nn.Linear(1000, self.img_feature_dim)

        # The linear layer that maps the LSTM with the 3 outputs
        self.last_layer = nn.Linear(self.img_feature_dim, self.number_of_outputs+1)


    def forward(self, x_in):

        base_out = self.base_model(x_in)
        base_out = torch.flatten(base_out, start_dim=1)
        output = self.last_layer(base_out)


        linear_output = output[:,:self.number_of_outputs]

        var = math.pi*nn.Sigmoid()(output[:,self.number_of_outputs:self.number_of_outputs+1])
        var = var.view(-1,1).expand(var.size(0), self.number_of_outputs)

        return linear_output,var


class EuclideanLoss(nn.Module):
    def __init__(self):
        super(EuclideanLoss, self).__init__()

    def forward(self, output_o, target_o, var_o):
        dlx = target_o[:,1]-output_o[:,1]
        dly = target_o[:,0]-output_o[:,0]
        drx = target_o[:,3]-output_o[:,3]
        dry = target_o[:,2]-output_o[:,2]
        dnx = target_o[:,5]-output_o[:,5]
        dny = target_o[:,4]-output_o[:,4]
        return torch.mean(dlx*dlx + dly*dly + drx*drx + dry*dry + dnx*dnx + dny*dny)


##############################################################################
# A PyTorch slice preparer

def prepare_slice_from_patch (patch):
	patch = patch/255.0
	patch = patch.transpose(2, 0, 1)

	slice = torch.from_numpy(patch[np.newaxis,:,:,:]).type(torch.FloatTensor)
	return slice

# A patch extractor

def extract_patch (image, ix, iy, iw, ih):
	if (ix < 0):
		iw += ix
		ix = 0

	if (iy < 0):
		ih += iy
		iy = 0

	patch = image[int(iy):int(iy+ih),int(ix):int(ix+iw)]
	return patch

############################################################################################

face_image_size = (100,150)

class Landmark3:

############################################################################################

	def __init__(self,checkpoint):
		self.thisnet = Coordinates()

		self.thisdevice = torch.device("cpu")
		self.thisnet.to(self.thisdevice)
		self.thisnet.load_state_dict(torch.load(checkpoint, map_location=self.thisdevice))
		self.thisnet.eval()

############################################################################################

	def predict (self,face_patch):
		face_slice = prepare_slice_from_patch(face_patch)
		landmarks, bias = self.thisnet(face_slice.to(self.thisdevice))

		for k, landmark in enumerate(landmarks):
			landmark = landmark.cpu().detach().numpy()
			#print('Predict',landmark)
			return (landmark[0],landmark[1],landmark[2],landmark[3],landmark[4],landmark[5])

		return (0,0,0,0,0,0)

	def process(self,image,face_rect):
		x = face_rect[0]
		y = face_rect[1]
		w = face_rect[2]
		h = face_rect[3]
		#print('Face Rect',x,y,w,h)
		face_patch = extract_patch(image,x,y,w,h)
		#print('FP before reshape',face_patch.shape)
		face_patch = cv2.resize(face_patch,face_image_size)
		lx,ly,rx,ry,nx,ny = self.predict(face_patch)
		#print('Landmark found (raw)',lx,ly,rx,ry)
		return (lx*w+x,ly*h+y,rx*w+x,ry*h+y,nx*w+x,ny*h+y)

############################################################################################
############################################################################################

if __name__ == '__main__':
	pass

############################################################################################
