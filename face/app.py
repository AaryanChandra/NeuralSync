from flask import Flask, render_template, Response, jsonify
import gunicorn
from camera import *
import time
from collections import deque
from collections import Counter as Counter
# FA jaaye maa chudane madarchod

app = Flask(__name__)

headings = ("Name","Album","Artist")
df1 = music_rec()
df1 = df1.head(15)

# Emotion timeline tracking
emotion_history = deque(maxlen=300)  # Store last 5 minutes at ~1 sample/second
session_start_time = time.time()

# Global camera instance for accessing tracker
camera_instance = None

@app.route('/')
def index():
    print(df1.to_json(orient='records'))
    return render_template('index.html', headings=headings, data=df1)

def gen(camera):
    global camera_instance, df1
    camera_instance = camera  # Store reference for API access
    while True:
        frame, df1 = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/t')
def gen_table():
    return df1.to_json(orient='records')

@app.route('/emotion')
def get_emotion():
    try:
        # Prefer stabilized backend status if available
        if isinstance(emotion_status, dict):
            idx = int(emotion_status.get('emotion_index', show_text[0]))
            detected = bool(emotion_status.get('face_detected', False))
            current = emotion_dict.get(idx, 'Neutral')
            
            # Add to history (sample every second)
            current_time = time.time()
            if len(emotion_history) == 0 or (current_time - emotion_history[-1]['timestamp']) >= 1.0:
                emotion_history.append({
                    'emotion': current,
                    'timestamp': current_time,
                    'faceDetected': detected
                })
            
            return jsonify({ 'emotion': current, 'faceDetected': detected })
        current = emotion_dict.get(show_text[0], 'Neutral')
    except Exception as e:
        current = 'Neutral'
    return jsonify({ 'emotion': current, 'faceDetected': True })

@app.route('/emotion_timeline')
def get_emotion_timeline():
    """Return emotion history for timeline visualization"""
    try:
        history_list = list(emotion_history)
        # Convert to relative timestamps (seconds from session start)
        timeline_data = []
        for entry in history_list:
            relative_time = entry['timestamp'] - session_start_time
            timeline_data.append({
                'time': round(relative_time, 1),
                'emotion': entry['emotion'],
                'faceDetected': entry.get('faceDetected', True)
            })
        return jsonify(timeline_data)
    except Exception as e:
        return jsonify([])

@app.route('/emotion_stats')
def get_emotion_stats():
    """Return session statistics"""
    try:
        if len(emotion_history) == 0:
            return jsonify({
                'dominantEmotion': 'Neutral',
                'sessionDuration': 0,
                'emotionCounts': {},
                'totalDetections': 0
            })
        
        # Calculate dominant emotion
        emotions = [e['emotion'] for e in emotion_history if e.get('faceDetected', True)]
        if emotions:
            emotion_counts = Counter(emotions)
            dominant_emotion = emotion_counts.most_common(1)[0][0]
            emotion_counts_dict = dict(emotion_counts)
        else:
            dominant_emotion = 'Neutral'
            emotion_counts_dict = {}
        
        # Calculate session duration
        session_duration = round(time.time() - session_start_time, 1)
        
        return jsonify({
            'dominantEmotion': dominant_emotion,
            'sessionDuration': session_duration,
            'emotionCounts': emotion_counts_dict,
            'totalDetections': len(emotion_history)
        })
    except Exception as e:
        return jsonify({
            'dominantEmotion': 'Neutral',
            'sessionDuration': 0,
            'emotionCounts': {},
            'totalDetections': 0
        })

@app.route('/multi_face_stats')
def get_multi_face_stats():
    """Return detailed statistics for all tracked faces"""
    try:
        global camera_instance
        if camera_instance is None:
            return jsonify({
                'face_count': 0,
                'faces': {},
                'group_mood': 'Neutral',
                'message': 'Camera not initialized'
            })
        
        tracker = camera_instance.get_tracker()
        faces_stats = tracker.get_all_faces_stats()
        group_mood = tracker.get_group_mood()
        
        # Convert face IDs to strings for JSON serialization
        faces_dict = {}
        for face_id, stats in faces_stats.items():
            faces_dict[str(face_id)] = {
                'dominant_emotion': stats['dominant_emotion'],
                'emotion_counts': stats['emotion_counts'],
                'total_detections': stats['total_detections'],
                'tracking_duration_frames': stats['tracking_duration'],
                'bbox': stats['bbox'],
                'centroid': stats['centroid']
            }
        
        return jsonify({
            'face_count': len(faces_stats),
            'faces': faces_dict,
            'group_mood': group_mood,
            'timestamp': time.time()
        })
    except Exception as e:
        return jsonify({
            'face_count': 0,
            'faces': {},
            'group_mood': 'Neutral',
            'error': str(e)
        })

@app.route('/face_tracking_info')
def get_face_tracking_info():
    """Return current face tracking information"""
    try:
        if isinstance(emotion_status, dict):
            tracked_faces = emotion_status.get('tracked_faces', {})
            face_count = emotion_status.get('face_count', 0)
            
            return jsonify({
                'face_count': face_count,
                'tracked_faces': tracked_faces,
                'face_detected': emotion_status.get('face_detected', False)
            })
        return jsonify({
            'face_count': 0,
            'tracked_faces': {},
            'face_detected': False
        })
    except Exception as e:
        return jsonify({
            'face_count': 0,
            'tracked_faces': {},
            'face_detected': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.debug = True
    app.run()
