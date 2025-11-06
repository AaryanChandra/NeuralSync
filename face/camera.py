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
import math
import os

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

face_cascade=cv2.CascadeClassifier(os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml"))
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
emotion_model.load_weights(os.path.join(BASE_DIR, 'model.h5'))

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0:"Angry",1:"Disgusted",2:"Fearful",3:"Happy",4:"Neutral",5:"Sad",6:"Surprised"}
music_dist={0:"angry.csv",1:"disgusted.csv",2:"fearful.csv",3:"happy.csv",4:"neutral.csv",5:"sad.csv",6:"surprised.csv"}
global last_frame1                                    
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1 
show_text=[4]
# Shared status for UI/endpoint
emotion_status = { 'emotion_index': 4, 'face_detected': False, 'updated_ms': 0 }

# Multi-face tracking system
class FaceTracker:
	"""Advanced multi-face tracking system with emotion tracking per face"""
	def __init__(self, max_disappeared=10, max_distance=100):
		self.next_id = 1
		self.faces = {}  # {face_id: {'centroid': (x, y), 'bbox': (x, y, w, h), 'emotions': deque, 'last_seen': frame_count}}
		self.disappeared = {}  # Track how long each face has been missing
		self.max_disappeared = max_disappeared
		self.max_distance = max_distance
		self.colors = [
			(0, 255, 0),    # Green
			(255, 0, 0),    # Blue
			(0, 0, 255),    # Red
			(255, 255, 0),  # Cyan
			(255, 0, 255),  # Magenta
			(0, 255, 255),  # Yellow
			(128, 0, 128),  # Purple
			(255, 165, 0)   # Orange
		]
	
	def _calculate_centroid(self, bbox):
		"""Calculate centroid of bounding box"""
		x, y, w, h = bbox
		return (x + w // 2, y + h // 2)
	
	def _calculate_iou(self, box1, box2):
		"""Calculate Intersection over Union (IoU) of two bounding boxes"""
		x1, y1, w1, h1 = box1
		x2, y2, w2, h2 = box2
		
		# Calculate intersection
		xi1 = max(x1, x2)
		yi1 = max(y1, y2)
		xi2 = min(x1 + w1, x2 + w2)
		yi2 = min(y1 + h1, y2 + h2)
		
		if xi2 <= xi1 or yi2 <= yi1:
			return 0.0
		
		inter_area = (xi2 - xi1) * (yi2 - yi1)
		box1_area = w1 * h1
		box2_area = w2 * h2
		union_area = box1_area + box2_area - inter_area
		
		if union_area == 0:
			return 0.0
		
		return inter_area / union_area
	
	def _calculate_distance(self, centroid1, centroid2):
		"""Calculate Euclidean distance between two centroids"""
		return math.sqrt((centroid1[0] - centroid2[0])**2 + (centroid1[1] - centroid2[1])**2)
	
	def update(self, detected_faces, frame_count):
		"""Update tracker with new detections"""
		# If no faces detected, increment disappeared counter
		if len(detected_faces) == 0:
			for face_id in list(self.disappeared.keys()):
				self.disappeared[face_id] += 1
				if self.disappeared[face_id] > self.max_disappeared:
					self._deregister(face_id)
			return self.faces
		
		# Calculate centroids for detected faces
		input_centroids = []
		for bbox in detected_faces:
			centroid = self._calculate_centroid(bbox)
			input_centroids.append((centroid, bbox))
		
		# If no existing faces, register all new ones
		if len(self.faces) == 0:
			for centroid, bbox in input_centroids:
				self._register(centroid, bbox, frame_count)
		else:
			# Match detected faces to existing tracked faces
			face_ids = list(self.faces.keys())
			face_centroids = [self.faces[fid]['centroid'] for fid in face_ids]
			
			# Calculate distance matrix
			D = []
			for i, (ic, ibbox) in enumerate(input_centroids):
				row = []
				for j, fc in enumerate(face_centroids):
					# Combined metric: IoU + distance
					iou = self._calculate_iou(ibbox, self.faces[face_ids[j]]['bbox'])
					dist = self._calculate_distance(ic, fc)
					# Normalize and combine (higher IoU and lower distance = better match)
					score = iou * 0.7 - (dist / self.max_distance) * 0.3
					row.append(score)
				D.append(row)
			
			# Greedy matching (assign best matches first)
			used_input_indices = set()
			used_face_ids = set()
			
			# Sort by best scores
			matches = []
			for i in range(len(input_centroids)):
				for j in range(len(face_ids)):
					if i not in used_input_indices and face_ids[j] not in used_face_ids:
						matches.append((D[i][j], i, face_ids[j]))
			
			matches.sort(reverse=True, key=lambda x: x[0])
			
			# Assign matches
			for score, input_idx, face_id in matches:
				if input_idx not in used_input_indices and face_id not in used_face_ids:
					if score > 0.3:  # Threshold for matching
						centroid, bbox = input_centroids[input_idx]
						self.faces[face_id]['centroid'] = centroid
						self.faces[face_id]['bbox'] = bbox
						self.faces[face_id]['last_seen'] = frame_count
						if face_id in self.disappeared:
							del self.disappeared[face_id]
						used_input_indices.add(input_idx)
						used_face_ids.add(face_id)
			
			# Register unmatched input faces as new
			for i, (centroid, bbox) in enumerate(input_centroids):
				if i not in used_input_indices:
					self._register(centroid, bbox, frame_count)
			
			# Handle disappeared faces
			for face_id in face_ids:
				if face_id not in used_face_ids:
					if face_id not in self.disappeared:
						self.disappeared[face_id] = 0
					self.disappeared[face_id] += 1
					if self.disappeared[face_id] > self.max_disappeared:
						self._deregister(face_id)
		
		return self.faces
	
	def _register(self, centroid, bbox, frame_count):
		"""Register a new face"""
		face_id = self.next_id
		self.next_id += 1
		self.faces[face_id] = {
			'centroid': centroid,
			'bbox': bbox,
			'emotions': deque(maxlen=30),  # Store last 30 emotion predictions
			'last_seen': frame_count,
			'first_seen': frame_count,
			'total_detections': 0
		}
	
	def _deregister(self, face_id):
		"""Remove a face from tracking"""
		if face_id in self.faces:
			del self.faces[face_id]
		if face_id in self.disappeared:
			del self.disappeared[face_id]
	
	def get_face_color(self, face_id):
		"""Get unique color for a face"""
		return self.colors[(face_id - 1) % len(self.colors)]
	
	def get_dominant_emotion(self, face_id):
		"""Get dominant emotion for a specific face"""
		if face_id not in self.faces or len(self.faces[face_id]['emotions']) == 0:
			return 'Neutral'
		emotions = list(self.faces[face_id]['emotions'])
		return Counter(emotions).most_common(1)[0][0]
	
	def add_emotion(self, face_id, emotion):
		"""Add emotion prediction for a face"""
		if face_id in self.faces:
			self.faces[face_id]['emotions'].append(emotion)
			self.faces[face_id]['total_detections'] += 1
	
	def get_all_faces_stats(self):
		"""Get statistics for all tracked faces"""
		stats = {}
		for face_id, face_data in self.faces.items():
			emotions = list(face_data['emotions'])
			if len(emotions) > 0:
				emotion_counts = dict(Counter(emotions))
				dominant = Counter(emotions).most_common(1)[0][0]
			else:
				emotion_counts = {}
				dominant = 'Neutral'
			
			stats[face_id] = {
				'dominant_emotion': dominant,
				'emotion_counts': emotion_counts,
				'total_detections': face_data['total_detections'],
				'tracking_duration': face_data['last_seen'] - face_data['first_seen'],
				'bbox': face_data['bbox'],
				'centroid': face_data['centroid']
			}
		return stats
	
	def get_group_mood(self):
		"""Calculate overall group mood from all faces"""
		if len(self.faces) == 0:
			return 'Neutral'
		
		all_emotions = []
		for face_data in self.faces.values():
			all_emotions.extend(list(face_data['emotions']))
		
		if len(all_emotions) == 0:
			return 'Neutral'
		
		return Counter(all_emotions).most_common(1)[0][0]


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
		# Initialize multi-face tracker
		self.face_tracker = FaceTracker(max_disappeared=10, max_distance=100)

	def _predict_emotion(self, gray, bbox):
		"""Predict emotion for a single face bounding box"""
		x, y, w, h = bbox
		roi_gray_frame = gray[y:y + h, x:x + w]
		cropped = cv2.resize(roi_gray_frame, (48, 48)).astype('float32') / 255.0
		cropped_img = np.expand_dims(np.expand_dims(cropped, -1), 0)
		prediction = emotion_model.predict(cropped_img, verbose=0)
		emotion_idx = int(np.argmax(prediction))
		return emotion_dict[emotion_idx], emotion_idx

	def get_frame(self):
		global last_frame1
		global df1
		global emotion_status
		global show_text
		# Read and preprocess frame
		image = self.cap.read()
		image = cv2.resize(image, (600, 500))
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# Throttle heavy ops: run detection/prediction every 3 frames
		if self.frame_count % 3 == 0:
			detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(60, 60))
			
			# Update face tracker
			tracked_faces = self.face_tracker.update(detected_faces, self.frame_count)
			
			# Predict emotions for each tracked face
			for face_id, face_data in tracked_faces.items():
				bbox = face_data['bbox']
				emotion_name, emotion_idx = self._predict_emotion(gray, bbox)
				self.face_tracker.add_emotion(face_id, emotion_name)
			
			# Determine primary emotion for music recommendations (use group mood or first face)
			if len(tracked_faces) > 0:
				group_mood = self.face_tracker.get_group_mood()
				# Map emotion name back to index
				for idx, name in emotion_dict.items():
					if name == group_mood:
						new_emotion_index = idx
						break
				else:
					new_emotion_index = 4  # Default to Neutral
				
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
			else:
				new_emotion_index = 4  # Neutral when no faces
				self.prediction_history.append(new_emotion_index)
			
			# Update face detected flag
			emotion_status['face_detected'] = (len(tracked_faces) > 0)
			emotion_status['face_count'] = len(tracked_faces)
			emotion_status['tracked_faces'] = {
				fid: {
					'emotion': self.face_tracker.get_dominant_emotion(fid),
					'bbox': tracked_faces[fid]['bbox']
				} for fid in tracked_faces.keys()
			}
		
		# Get current tracked faces for drawing (even if not updated this frame)
		tracked_faces = self.face_tracker.faces

		# Draw overlays for each tracked face with unique colors and IDs
		for face_id, face_data in tracked_faces.items():
			x, y, w, h = face_data['bbox']
			color = self.face_tracker.get_face_color(face_id)
			dominant_emotion = self.face_tracker.get_dominant_emotion(face_id)
			
			# Draw bounding box with unique color
			cv2.rectangle(image, (x, y-50), (x+w, y+h+10), color, 2)
			
			# Draw only emotion label (without Person ID)
			label = dominant_emotion
			cv2.putText(image, label, (x+5, y-60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

		self.frame_count += 1
		# Publish status for endpoint consumers
		emotion_status['emotion_index'] = int(self.current_emotion_index)
		emotion_status['updated_ms'] = int(time.time() * 1000)

		# Output frame and latest recommendations
		last_frame1 = image.copy()
		ret, jpeg = cv2.imencode('.jpg', last_frame1)
		df1 = self.df_cache
		return jpeg.tobytes(), df1
	
	def get_tracker(self):
		"""Get the face tracker instance for external access"""
		return self.face_tracker

def music_rec():
	# print('---------------- Value ------------', music_dist[show_text[0]])
	csv_path = os.path.join(BASE_DIR, music_dist[show_text[0]])
	df = pd.read_csv(csv_path)
	df = df[['Name','Album','Artist']]
	df = df.head(15)
	return df
