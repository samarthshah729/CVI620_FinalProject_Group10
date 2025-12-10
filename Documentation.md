# Behavioral Cloning – Self-Driving Car Project

**Computer Vision (CVI620) – Final Project**

## 1. Overview

This project implements a behavioral cloning system to autonomously steer a vehicle in the Udacity Self-Driving Car Simulator. A convolutional neural network (CNN) is trained on recorded driving data and used to predict steering angles in real time.

## 2. Project Structure

```
FinalProject/
│
├── data/
│   ├── IMG/
│   ├── driving_log.csv
│
├── simulator/
│   └── Default Windows desktop 64-bit.exe
│
├── train.py
├── TestSimulation.py
├── model.h5
├── loss_plot.png
├── tf_temp/
└── README.md
```

## 3. Environment Setup

### 3.1 Required Versions (Critical for Simulator Compatibility)

```bash
pip install python-socketio==3.1.2
pip install python-engineio==3.9.3
pip install flask-socketio==3.0.2
pip install eventlet==0.30.2
```

### 3.2 Additional Dependencies

```bash
pip install tensorflow
pip install opencv-python
pip install numpy pandas matplotlib pillow scikit-learn
```

## 4. Training Instructions

### 4.1 Dataset Placement

Ensure your dataset is placed in the following structure:

```
data/
   IMG/
   driving_log.csv
```

### 4.2 Run Training

```bash
python train.py
```

This script performs:

- Data loading
- Path correction
- Preprocessing
- Augmentation
- Model creation (Nvidia CNN)
- 25-epoch training
- Saves `model.h5`
- Generates `loss_plot.png`

## 5. Running the Autonomous Vehicle

### 5.1 Steps

1. Launch Udacity Simulator
2. Select **Autonomous Mode**
3. In terminal run:

```bash
python TestSimulation.py
```

If successful, you will see:

```
Model loaded. Waiting for simulator...
Connected
Receiving telemetry...
```

The vehicle will begin autonomous driving.

## 6. Challenges Faced

### 6.1 Dependency and Version Conflicts

**Problem**

The simulator did not send telemetry data due to incompatible versions of:

- `python-socketio`
- `python-engineio`
- `flask-socketio`
- `eventlet`

The result was that the car did not move because no image frames or speed data were transmitted.

**Fix**

Installed legacy versions:

```bash
python-socketio==3.1.2  
python-engineio==3.9.3  
flask-socketio==3.0.2  
eventlet==0.30.2
```

These versions restored full communication with the simulator.

### 6.2 TensorFlow Temporary Directory Error

**Problem**

TensorFlow attempted to write temporary files to system-restricted directories, causing crashes.

**Fix**

Created a dedicated project-level temporary folder and forced TensorFlow to use it:

```python
os.environ['TMPDIR'] = custom_temp_dir
tempfile.tempdir = custom_temp_dir
```

### 6.3 Preprocessing Mismatch

**Problem**

Training and inference used slightly different preprocessing steps. This caused unstable steering, zig-zagging, and drifting off the road.

**Fix**

Made preprocessing identical in both scripts:

1. Crop: `img[60:135, :, :]`
2. RGB → YUV
3. Gaussian Blur
4. Resize (200, 66)
5. Normalize

### 6.4 Overfitting and Poor Driving Performance

**Problem**

Initial model overfit small datasets and performed poorly in simulation.

**Fix**

- Added teammate's high-quality dataset
- Increased epochs to 25
- Added augmentation (flip, brightness, shift)
- Balanced left-right turns

This resulted in smoother and stable driving.

## 7. Changes Made to TestSimulation.py

### 7.1 Preprocessing Updated

Matched exactly with training preprocessing to ensure consistency.

### 7.2 Prediction Extraction Fixed

Updated steering prediction to:

```python
steering = float(model.predict(image)[0][0])
```

### 7.3 Required Event Handlers Added

```python
@sio.on('manual')
def manual(sid, data):
    pass
```

### 7.4 Throttle Stabilized

Dynamic throttle was replaced with a stable constant:

```python
throttle = 0.25
```

### 7.5 SocketIO Compatibility Adjustments

Ensured all handlers and imports follow the legacy WebSocket protocol expected by the simulator.

### 7.6 Debug Logging Added

Telemetry logging made it easier to confirm data flow and troubleshoot when the car did not move.

## 8. Work Distribution

### Samarth Kamlesh Shah
- Environment setup and dependency debugging
- Wrote preprocessing, augmentation, and full training pipeline
- Debugged and rewrote TestSimulation.py
- Integrated teammate datasets
- Trained final model and produced simulation demo
- Handled majority of testing and troubleshooting

### Ved Snehalkumar Patel
- Recorded additional training data
- Organized dataset structure
- Assisted in identifying overfitting issues
- Participated in early simulation testing

### Yash Hiteshkumar Shah
- Ran multiple autonomous simulation tests
- Verified correctness of preprocessing and augmentation
- Helped debug simulator-Python communication
- Assisted in documentation and explanation

## 9. Deliverables Checklist

| Deliverable | Status |
|------------|--------|
| Training script (train.py) | Completed |
| Inference script (TestSimulation.py) | Completed |
| Preprocessing and augmentation | Completed |
| Trained model (model.h5) | Completed |
| Simulation video | Completed |
| README documentation | Completed |
| Git repository | Organized |

## 10. How to Run the Entire Project

### Train the Model

```bash
python train.py
```

### Run Autonomous Mode

```bash
python TestSimulation.py
```

Vehicle will automatically begin driving once simulator connects.

