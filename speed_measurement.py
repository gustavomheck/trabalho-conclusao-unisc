from collections import OrderedDict
from objecttracker import ObjectTracker
from trackableobject import TrackableObject
from imutils.video import FileVideoStream
from imutils.video import FPS
import tensorflow as tf
import tensornets as tn
import numpy as np
import argparse
import imutils
import cv2
import math
import warnings

# Ignore deprecation warnings
warnings.filterwarnings("ignore")

# Window title
TITLE = "output"

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", help="Path to input video")
ap.add_argument("-c", "--confidence", type=float, help="Minimum confidence")
ap.add_argument("-d", "--distance", type=int, help="Maximum distance to associate two objects")
ap.add_argument("-o", "--output", action="store_true", help="Save each frame on disk")
args = vars(ap.parse_args())

# Path to input video file
VIDEO_NAME = args["input"] if args["input"] is not None else "videos/video2.mp4"

# Get the framerate
vc = cv2.VideoCapture(VIDEO_NAME)
framerate = vc.get(cv2.CAP_PROP_FPS)
vc.release()

# Minimun confidence of the object detection
CONFIDENCE = args["confidence"] if args["confidence"] is not None else 0.7

# Maximum distance to associate two objects
DISTANCE = args["distance"] if args["distance"] is not None else 30

# Debug mode means each frame will be saved on disk
DEBUG = not args["output"]

# Define the width and height of the video
W, H = 384, 288

# Define the top and bottom margins
TOP_MARGIN, BOTTOM_MARGIN = (H * 10) // 100, H - ((H * 10) // 100)

# Allow dynamic GPU RAM allocation
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

# Define tensor shape, model and classes that will be detected
# Detection algorithm is YOLOv3 trained on the VOC database
CLASSES = {2: "car", 3: "bike", 5: "bus", 7: "truck"}
inputs = tf.compat.v1.placeholder(tf.float32, [None, W, H, 3])
model = tn.YOLOv3COCO(inputs, tn.TinyDarknet19)

# Initialize the object tracker
tracker = ObjectTracker(CLASSES, framerate, TOP_MARGIN, BOTTOM_MARGIN, CONFIDENCE, DISTANCE, DEBUG)

with tf.compat.v1.Session(config=config) as sess:
    print("[INFO] Running pre-trained model...")
    sess.run(model.pretrained())

    # Create and resize the window used to show the frames
    cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TITLE, W, H)

    # Start the video and FPS counter
    fvs = FileVideoStream(VIDEO_NAME).start()    
    fps = FPS().start()

    # Initialize variables to identify the frames and count them
    frame_id, frame_counter = 0, 2

    while fvs.more():
        frame = fvs.read()
        if frame is None:
            # We reached the end of the video
            continue

        # Quit when 'Q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Increment the frame ID
        frame_id += 1

        if frame_counter < 2:
            frame_counter += 1
            continue

        # Reset frame counter to 1
        frame_counter = 1

        # Resize the frame so we get better performance
        # when running the object detection
        frame = cv2.resize(frame, (W, H))
        img = np.array(frame).reshape(-1, H, W, 3)

        # Run the object detection and get the predictions
        preds = sess.run(model.preds, {inputs: model.preprocess(img)}) 
        boxes = model.get_boxes(preds, img.shape[1:3])
        predictions = np.array(boxes)

        # Update the object tracker
        tracker.update(frame, frame_id, predictions)

        # Show the frame on a window and update de FPS counter
        cv2.imshow(TITLE, frame)
        fps.update()
    
    # Stop FPS counter and video
    fps.stop()
    fvs.stop()

    # Log the elapsed time and average FPS
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# End of execution
cv2.destroyAllWindows()