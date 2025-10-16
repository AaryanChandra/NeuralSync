import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pandastable import Table, TableModel
from tensorflow.keras.preprocessing import image
import datetime
from threading import Thread
from collections import deque, Counter
# from Spotipy import *  
import time
import pandas as pd
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
ds_factor=0.6

emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('model.h5')

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0:"Angry",1:"Disgusted",2:"Fearful",3:"Happy",4:"Neutral",5:"Sad",6:"Surprised"}
music_dist={0:"angry.csv",1:"disgusted.csv",2:"fearful.csv",3:"happy.csv",4:"neutral.csv",5:"sad.csv",6:"surprised.csv"}
global last_frame1                                    
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1 
show_text=[4]
# Shared status for UI/endpoint
emotion_status = { 'emotion_index': 4, 'face_detected': False, 'updated_ms': 0 }


''' Class for calculating FPS while streaming. Used this to check performance of using another thread for video streaming '''
class FPS:
	def __init__(self):
		# store the start time, end time, and total number of frames
		# that were examined between the start and end intervals
		self._start = None
		self._end = None
		self._numFrames = 0
	def start(self):
		# start the timer
		self._start = datetime.datetime.now()
		return self
	def stop(self):
		# stop the timer
		self._end = datetime.datetime.now()
	def update(self):
		# increment the total number of frames examined during the
		# start and end intervals
		self._numFrames += 1
	def elapsed(self):
		# return the total number of seconds between the start and
		# end interval
		return (self._end - self._start).total_seconds()
	def fps(self):
		# compute the (approximate) frames per second
		return self._numFrames / self.elapsed()


''' Class for using another thread for video streaming to boost performance '''
class WebcamVideoStream:
    	
		def __init__(self, src=0):
			self.stream = cv2.VideoCapture(src,cv2.CAP_DSHOW)
			# Reduce resolution to lower CPU usage
			self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
			self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
			(self.grabbed, self.frame) = self.stream.read()
			self.stopped = False

		def start(self):
				# start the thread to read frames from the video stream
			Thread(target=self.update, args=()).start()
			return self
			
		def update(self):
			# keep looping infinitely until the thread is stopped
			while True:
				# if the thread indicator variable is set, stop the thread
				if self.stopped:
					return
				# otherwise, read the next frame from the stream
				(self.grabbed, self.frame) = self.stream.read()

		def read(self):
			# return the frame most recently read
			return self.frame
		def stop(self):
			# indicate that the thread should be stopped
			self.stopped = True

''' Class for reading video stream, generating prediction and recommendations '''
class VideoCamera(object):
	def __init__(self):
		# Reuse a single webcam stream to avoid reopening every frame
		self.cap = WebcamVideoStream(src=0).start()
		self.frame_count = 0
		self.last_faces = []
		self.current_emotion_index = show_text[0]
		self.df_cache = music_rec()
		# Keep a short history of predictions to smooth jitter
		self.prediction_history = deque(maxlen=9)

	def _predict_emotion(self, gray, faces):
		# Predict on the first face detected to reduce compute
		if len(faces) == 0:
			return self.current_emotion_index
		x, y, w, h = faces[0]
		roi_gray_frame = gray[y:y + h, x:x + w]
		cropped = cv2.resize(roi_gray_frame, (48, 48)).astype('float32') / 255.0
		cropped_img = np.expand_dims(np.expand_dims(cropped, -1), 0)
		prediction = emotion_model.predict(cropped_img, verbose=0)
		return int(np.argmax(prediction))

	def get_frame(self):
		global last_frame1
		global df1
		global emotion_status
		# Read and preprocess frame
		image = self.cap.read()
		image = cv2.resize(image, (600, 500))
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# Throttle heavy ops: run detection/prediction every 3 frames
		if self.frame_count % 3 == 0:
			faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(60, 60))
			self.last_faces = faces
			new_emotion_index = self._predict_emotion(gray, faces)
			self.prediction_history.append(new_emotion_index)
			# Find the most common prediction in history
			mode_emotion, mode_count = None, 0
			if len(self.prediction_history) > 0:
				counts = Counter(self.prediction_history)
				mode_emotion, mode_count = counts.most_common(1)[0]
			# Require majority stability before switching
			if mode_emotion is not None and mode_count >= max(3, len(self.prediction_history)//2 + 1):
				if mode_emotion != self.current_emotion_index:
					self.current_emotion_index = mode_emotion
					show_text[0] = mode_emotion
					# Only recompute recommendations when emotion changes
					self.df_cache = music_rec()
			# Update face detected flag
			emotion_status['face_detected'] = (len(faces) > 0)
		else:
			faces = self.last_faces

		# Draw overlays using current emotion
		for (x, y, w, h) in faces:
			cv2.rectangle(image, (x, y-50), (x+w, y+h+10), (0, 255, 0), 2)
			cv2.putText(image, emotion_dict[self.current_emotion_index], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

		self.frame_count += 1
		# Publish status for endpoint consumers
		emotion_status['emotion_index'] = int(self.current_emotion_index)
		emotion_status['updated_ms'] = int(time.time() * 1000)

		# Output frame and latest recommendations
		last_frame1 = image.copy()
		ret, jpeg = cv2.imencode('.jpg', last_frame1)
		df1 = self.df_cache
		return jpeg.tobytes(), df1

def music_rec():
	# print('---------------- Value ------------', music_dist[show_text[0]])
	df = pd.read_csv(music_dist[show_text[0]])
	df = df[['Name','Album','Artist']]
	df = df.head(15)
	return df
