import cv2
import numpy as np

############################################################################################
# Parameters
# -------------------------------------------------------------------

CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
IMG_WIDTH = 416
IMG_HEIGHT = 416

############################################################################################
# Get the names of the output layers
def get_outputs_names(net):
	# Get the names of all the layers in the network
	layers_names = net.getLayerNames()

	# Get the names of the output layers, i.e. the layers with unconnected outputs
	uco = net.getUnconnectedOutLayers()
	return [layers_names[i - 1] for i in uco]

############################################################################################

def post_process(frame, outs, conf_threshold, nms_threshold):
	frame_height = frame.shape[0]
	frame_width = frame.shape[1]

	# Scan through all the bounding boxes output from the network and keep only the ones with high confidence scores.
	# Assign the box's class label as the class with the highest score.
	confidences = []
	boxes = []
	final_boxes = []
	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence > conf_threshold:
				center_x = int(detection[0] * frame_width)
				center_y = int(detection[1] * frame_height)
				width = int(detection[2] * frame_width)
				height = int(detection[3] * frame_height)
				left = int(center_x - width / 2)
				top = int(center_y - height / 2)
				confidences.append(float(confidence))
				boxes.append([left, top, width, height])

	# Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences.
	indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

	for i in indices:
		box = boxes[i]
		left = box[0]
		top = box[1]
		width = box[2]
		height = box[3]
		final_boxes.append(box)
		left, top, right, bottom = refined_box(left, top, width, height)
		# draw_predict(frame, confidences[i], left, top, left + width, top + height)
		# draw_predict(frame, confidences[i], left, top, right, bottom)

	return final_boxes

############################################################################################

def refined_box(left, top, width, height):
	right = left + width
	bottom = top + height

	original_vert_height = bottom - top
	top = int(top + original_vert_height * 0.15)
	bottom = int(bottom - original_vert_height * 0.05)

	margin = ((bottom - top) - (right - left)) // 2
	left = left - margin if (bottom - top - right + left) % 2 == 0 else left - margin - 1

	right = right + margin

	return left, top, right, bottom

############################################################################################

def startup():
	net = cv2.dnn.readNetFromDarknet('config/yolov3-face.cfg', 'pretrained/yolov3-wider_16000.weights')
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
	return net

############################################################################################

def process(net,frame):
	# Create a 4D blob from a frame.
	blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT), [0, 0, 0], 1, crop=False)

	# Sets the input to the network
	net.setInput(blob)

	# Runs the forward pass to get output of the output layers
	outs = net.forward(get_outputs_names(net))

	# Remove the bounding boxes with low confidence
	faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
	return faces

############################################################################################

def best_face(net,frame):
	faces_found = process(net,frame)

	if len(faces_found) == 0:
		return (False,(0,0,0,0))
	else:
		return (True,faces_found[0])

def face_around(net,frame,point):
	faces_found = process(net,frame)

	px = point[0]
	py = point[1]
	for (ix,iy,iw,ih) in faces_found:
		if (px >= ix) and (px < ix+iw) and (py >= iy) and (py < iy+ih):
			return (True,(ix,iy,iw,ih))

	return (False,(0,0,0,0))

############################################################################################
