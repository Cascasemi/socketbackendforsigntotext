import os
from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
import pickle
import cv2
import mediapipe as mp
import numpy as np
import eventlet
import logging
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize eventlet
eventlet.monkey_patch()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure Socket.IO
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
labels_dict = {
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
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    with mp_hands.Hands(
            static_image_mode=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

        while True:
            data_aux = []
            x_ = []
            y_ = []

            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to grab frame")
                break

            frame = cv2.flip(frame, 1)
            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks and model:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    for landmark in hand_landmarks.landmark:
                        x = landmark.x
                        y = landmark.y
                        x_.append(x)
                        y_.append(y)

                    for landmark in hand_landmarks.landmark:
                        x = landmark.x
                        y = landmark.y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10
                    x2 = int(max(x_) * W) - 10
                    y2 = int(max(y_) * H) - 10

                    try:
                        prediction = model.predict([np.asarray(data_aux)])
                        prediction_proba = model.predict_proba([np.asarray(data_aux)])
                        confidence = max(prediction_proba[0])
                        predicted_character = labels_dict[int(prediction[0])]

                        socketio.emit('prediction', {
                            'text': predicted_character,
                            'confidence': float(confidence)
                        })

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                        cv2.putText(
                            frame,
                            f"{predicted_character} ({confidence * 100:.2f}%)",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.3,
                            (0, 0, 0),
                            3,
                            cv2.LINE_AA
                        )
                    except Exception as e:
                        logger.error(f"Prediction error: {e}")

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")
    emit('connection_response', {'data': 'Connected'})


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