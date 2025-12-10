import socketio
import eventlet
import eventlet.wsgi
from flask import Flask
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import cv2
from tensorflow.keras.models import load_model

# Create SocketIO server
sio = socketio.Server()
app = Flask(__name__)

model = None

def preprocess(img):
    """Preprocess the image like in training"""
    img = img[60:135, :, :]                     # crop
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # color convert
    img = cv2.GaussianBlur(img, (3,3), 0)       # blur
    img = cv2.resize(img, (200,66))             # resize
    img = img.astype(np.float32) / 255.0        # normalize
    return img

@sio.on('connect')
def connect(sid, environ):
    print("  Simulator connected:", sid)
    send_control(0, 0)

@sio.on('telemetry')
def telemetry(sid, data):
    if data is None:
        print(" No telemetry received.")
        return
    
    # Read data
    speed = float(data["speed"])
    
    img_str = data["image"]
    image = Image.open(BytesIO(base64.b64decode(img_str)))
    image = np.asarray(image)
    
    # Preprocess image
    image = preprocess(image)
    image = np.expand_dims(image, axis=0)
    
    # Predict steering
    steering = float(model.predict(image)[0][0])
    
    # Simple constant throttle
    throttle = 0.09
    
    print(f"STEERING={steering:.3f} | THROTTLE={throttle:.3f} | SPEED={speed:.3f}")
    
    send_control(steering, throttle)

def send_control(steering, throttle):
    sio.emit("steer", data={
        "steering_angle": str(steering),
        "throttle": str(throttle)
    })

if __name__ == "__main__":
    print("Loading model...")
    model = load_model("model.h5")
    print("Model loaded. Waiting for simulator...")

    # Wrap Flask app with SocketIO middleware
    app = socketio.Middleware(sio, app)

    # Start server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
