# MUST BE FIRST - Eventlet monkey patching
import eventlet

eventlet.monkey_patch()

# Now regular imports
import os
import pickle
import numpy as np
import logging
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
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
    return render_template('index.html')


@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")


@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")


@socketio.on('frame_data')
def handle_frame_data(data):
    if model is None:
        socketio.emit('prediction', {'error': 'Model not loaded'})
        return

    try:
        # Convert received data to numpy array
        frame_data = np.frombuffer(data['frame'], dtype=np.uint8)
        landmarks = data['landmarks']

        # Process landmarks and make prediction
        data_aux = []
        x_coords = [lm['x'] for lm in landmarks]
        y_coords = [lm['y'] for lm in landmarks]

        for lm in landmarks:
            data_aux.append(lm['x'] - min(x_coords))
            data_aux.append(lm['y'] - min(y_coords))

        prediction = model.predict([np.asarray(data_aux)])
        confidence = max(model.predict_proba([np.asarray(data_aux)])[0])
        char = LABELS[int(prediction[0])]

        socketio.emit('prediction', {
            'text': char,
            'confidence': float(confidence)
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        socketio.emit('prediction', {'error': str(e)})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app,
                 host='0.0.0.0',
                 port=port,
                 debug=False,
                 use_reloader=False)