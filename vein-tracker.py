import cv2 #OpenCV for video capture and display
import mediapipe as mp # For hand tracking 
import os 
from mediapipe.tasks import python # Import python API for MediaPipe tasks
from mediapipe.tasks.python import vision # Vision tasks 


#setup mediapipe hand tracking model
model_path = '/absolute/path/to/gesture_recognizer.task'


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the live stream mode:
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('hand landmarker result: {}'.format(result))

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='/path/to/model.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)
with HandLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. Use it here.


#open cv stuff - Initializ webcam capture 
cap = cv2.VideoCapture(0) 

# Getting dimensions of the video frames 
Frame_width = int(cap.get(3)) # Width of the frame 
Frame_height = int(cap.get(4)) # Height of the frame

# Main video processing loop 
while True: 
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow("Webcam", frame)









    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()    # Convert the frame to grayscale
cv2.destroyAllWindows()