from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def detect_and_predict_mask(f, ff, fff):
	(h, w) = f.shape[:2]
	blob = cv2.dnn.blobFromImage(f, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	
	ff.setInput(blob)
	detections = ff.forward()

	faces = []
	locs = []
	preds = []

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > args["confidence"]:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			face = f[startY:endY, startX:endX]
			if face.any():
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)

				faces.append(face)
				locs.append((startX, startY, endX, endY))

	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = fff.predict(faces, batch_size=32)

	return (locs, preds)

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="fd",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="d.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


prototxtPath = os.path.sep.join([args["face"], "video.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"1.caffemodel"])
ff = cv2.dnn.readNet(prototxtPath, weightsPath)



fff = load_model(args["model"])


vs = VideoStream(src=0).start()
time.sleep(2.0)


while True:
	f = vs.read()
	f = imutils.resize(f, width=400)
	(locs, preds) = detect_and_predict_mask(f, ff, fff)

	for (box, pred) in zip(locs, preds):
		
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 255) if label == "Mask" else (255, 0, 255)
			
		cv2.putText(f, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(f, (startX, startY), (endX, endY), color, 2)

	cv2.imshow("f", f)
	key = cv2.waitKey(1) & 0xFF

