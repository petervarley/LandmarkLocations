import Landmark3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import sys
import os
import cv2

#####################################################################################

class loader(Dataset):
  def __init__(self, path):
    with open(path) as f:
      self.lines = f.readlines()

  def __len__(self):
    return len(self.lines)

  def __getitem__(self, idx):
    line = self.lines[idx]
    line = line.strip().split(" ")

    #print('getitem line',line)
    gt = (float(line[1])/100,float(line[2])/150,float(line[3])/100,float(line[4])/150,float(line[5])/100,float(line[6])/150)
    label = torch.from_numpy(np.array(gt).astype(float)).type(torch.FloatTensor)

    try:
        img_filename = line[0]
        #print('Reading:',img_filename)
        eimg = cv2.imread(img_filename)/255.0
        img = torch.from_numpy(eimg.transpose(2, 0, 1)).type(torch.FloatTensor)
    except TypeError as e:
        print('Type error for ',line[0])
        raise e

    return img, label

#####################################################################################

def train_3_landmarks(modelname,labelpath):

	# i represents the i-th folder used as the test set.
	savepath = 'checkpoint'
	if not os.path.exists(savepath):
		os.makedirs(savepath)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	print("Read data",labelpath)
	dataset = loader(labelpath)
	print(f"[Read Data]: Total num: {len(dataset)}")
	print(f"[Read Data]: Label path: {labelpath}")
	dataset = DataLoader(dataset, batch_size=80, shuffle=True, num_workers=4)

	print("Model building")
	net = Landmark3.Coordinates()
	net.train()
	net.to(device)

	print("optimizer building")
	loss_op = Landmark3.EuclideanLoss().cuda()
	base_lr = 0.0001

	decaysteps = 5000
	decayratio = 1
	epochs = 200

	optimizer = optim.Adam(net.parameters(),lr=base_lr, betas=(0.9,0.95))
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decaysteps, gamma=decayratio)

	print("Training")
	length = len(dataset)
	total = length * epochs
	cur = 0
	timebegin = time.time()
	with open(os.path.join(savepath, f'{modelname}_train_log.csv'), 'w') as outfile:
		for epoch in range(1, epochs+1):
			for i, (data, label) in enumerate(dataset):

				# Acquire data
				label = label.to(device)

				# forward
				gaze, gaze_bias = net(data.to(device))

				# loss calculation
				loss = loss_op(gaze, label, gaze_bias)
				optimizer.zero_grad()

				# backward
				loss.backward()
				optimizer.step()
				scheduler.step()
				cur += 1

				# print logs
				if i % 20 == 0:
					timeend = time.time()
					resttime = (timeend - timebegin)/cur * (total-cur)/3600
					print(f"[{epoch}/{epochs}]: [{i}/{length}] loss:{loss} lr:{base_lr}, rest time:{resttime:.2f}h")
					outfile.write(f"{epoch+(i/length):.3},{loss}\n")
					sys.stdout.flush()
					outfile.flush()

			if epoch % 10 == 0:
				torch.save(net.state_dict(), os.path.join(savepath, f"Iter_{epoch}_{modelname}.pt"))

#####################################################################################

if __name__ == "__main__":
	print("Training")
	torch.manual_seed(3)
	train_3_landmarks(sys.argv[1],sys.argv[2])

#####################################################################################
# Usage:
# python python train3landmarks.py Landmarks3 lists/landmark_train.txt
