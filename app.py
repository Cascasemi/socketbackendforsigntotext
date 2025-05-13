# MUST BE FIRST - Eventlet monkey patching
import eventlet

eventlet.monkey_patch()

# Now regular imports
import os
from flask import Flask, Response, jsonify, request
from flask_socketio import SocketIO
import pickle
import cv2
import mediapipe as mp
import numpy as np
import logging
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask and SocketIO
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app,
                    cors_allowed_origins="*",
                    async_mode='eventlet',
                    logger=True,
                    engineio_logger=True)

# Load model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# Labels dictionary
LABELS = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Hello',
    27: 'Done', 28: 'Thank You', 29: 'I Love you', 30: 'Sorry', 31: 'Please',
    32: 'You are welcome.'
}


@app.route('/')
def index():
    return jsonify({"status": "success", "message": "Sign2Text API is running"})


@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"})


def generate_frames():
    cap = cv2.VideoCapture(0)
    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks and model:
            for landmarks in results.multi_hand_landmarks:
                # Process landmarks and make prediction
                data_aux = []
                x_coords = []
                y_coords = []

                for landmark in landmarks.landmark:
                    x_coords.append(landmark.x)
                    y_coords.append(landmark.y)

                for landmark in landmarks.landmark:
                    data_aux.append(landmark.x - min(x_coords))
                    data_aux.append(landmark.y - min(y_coords))

                try:
                    prediction = model.predict([np.asarray(data_aux)])
                    confidence = max(model.predict_proba([np.asarray(data_aux)])[0])
                    char = LABELS[int(prediction[0])]

                    socketio.emit('prediction', {
                        'text': char,
                        'confidence': float(confidence)
                    })

                    # Draw on frame
                    h, w, _ = frame.shape
                    x1, y1 = int(min(x_coords) * w) - 10, int(min(y_coords) * h) - 10
                    x2, y2 = int(max(x_coords) * w) - 10, int(max(y_coords) * h) - 10

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{char} ({confidence * 100:.1f}%)",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                except Exception as e:
                    logger.error(f"Prediction error: {e}")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")


@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app,
                 host='0.0.0.0',
                 port=port,
                 debug=False,
                 use_reloader=False)