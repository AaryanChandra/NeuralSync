from flask import Flask, render_template, Response, jsonify
import gunicorn
from camera import *

app = Flask(__name__)

headings = ("Name","Album","Artist")
df1 = music_rec()
df1 = df1.head(15)
@app.route('/')
def index():
    print(df1.to_json(orient='records'))
    return render_template('index.html', headings=headings, data=df1)

def gen(camera):
    while True:
        global df1
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
        if 'emotion_status' in globals() and isinstance(emotion_status, dict):
            idx = int(emotion_status.get('emotion_index', show_text[0]))
            detected = bool(emotion_status.get('face_detected', False))
            current = emotion_dict.get(idx, 'Neutral')
            return jsonify({ 'emotion': current, 'faceDetected': detected })
        current = emotion_dict.get(show_text[0], 'Neutral')
    except Exception:
        current = 'Neutral'
    return jsonify({ 'emotion': current, 'faceDetected': True })

if __name__ == '__main__':
    app.debug = True
    app.run()
