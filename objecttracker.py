from collections import OrderedDict
from trackableobject import TrackableObject
from trackingdebugger import TrackingDebugger
from faker import Factory
import numpy as np
import cv2
import math
import time

class ObjectTracker:
    def __init__(self, classes, framerate, top_margin=0, bottom_margin=0, confidence=0.7, max_distance=30, debug=False):
        # Initialize the next unique object ID 
        self.next_object_id = 1
        # Classes of objects that will be tracked
        self.classes = classes
        # Initialize an ordered dictionary used to keep track of classes
        # and the objects currently tracked that belong to that class        
        self.objects = OrderedDict()
        # The input's video framerate
        self.framerate = 1.0 / framerate
        # Store the top and bottom margins. Objects inside this region
        # will be ignored or unregistered
        self.top_margin = top_margin
        self.bottom_margin = bottom_margin
        # Mininum confidence of the object detection
        self.confidence = confidence
        # Store the maximum distance between centroids to associate
		# an object. If the distance is larger than this maximum
        # it means that it is a new object
        self.max_distance = max_distance        
        # Should save each frame and and prediction on disk
        self.debug = debug
        # Factory that generate colors to identify each object
        self.fake = Factory().create()
        # Startup time
        self.startup_time = time.strftime("%d%m%H%M%S")
    
    def update(self, frame, frame_id, predictions):
        # Store the matching objects and the new objects 
        # that sould be registered
        matches = {}
        to_register = []

        for obj_class in self.classes:
            for i in range(0, len(predictions[obj_class])):
                pred = predictions[obj_class][i]
                x1 = int(pred[0])
                y1 = int(pred[1])
                x2 = int(pred[2])
                y2 = int(pred[3])
                confidence = pred[4]
                if y1 < self.top_margin or y2 > self.bottom_margin or confidence < self.confidence:
                    continue

                # Find the center of the bounding box (centroid)
                (a, b) = (x1 + x2) // 2, (y1 + y2) // 2
                match = 0
                for obj_id in self.objects:
                    last_pos = self.objects[obj_id].get_last_position()
                    (c, d) = last_pos[1]
                    dist = math.sqrt(pow(a - c, 2) + pow(b - d, 2))
                    if dist <= self.max_distance:
                        # If the distance between two centroids is smaller
                        # than the max_distance then we assume they are
                        # the same object
                        if last_pos[7] != frame_id:
                            # The object also cannot have been already matched
                            # in this frame
                            match = obj_id
                            matches[obj_id] = None
                            break

                if match == 0 or match != -1:
                    obj = None
                    obj_id = 0
                    color = None
                    if match == 0:
                        # The centroid wasn't close enough to any tracked object, so it will be
                        # registered as a new object
                        color = tuple(np.random.random(size=3) * 256)
                        pos = (obj_class, (a, b), x1, y1, x2, y2, confidence, frame_id)
                        obj_id = self.next_object_id
                        obj = TrackableObject(obj_id, color, pos)
                        self.objects[obj_id] = obj
                        self.next_object_id += 1
                    elif match != -1:
                        # Append to the current TrackleObject the new position of the object
                        obj_id = match
                        obj = self.objects[obj_id]
                        obj.reset_missing_count()
                        obj.append_position((obj_class, (a, b), x1, y1, x2, y2, confidence, frame_id))
                        color = obj.get_color()

                    # Draw the bounding box
                    last_pos = obj.get_last_position()
                    label = "{} {}".format(self.classes[last_pos[0]], obj_id)
                    white = (255, 255, 255)
                    line = cv2.LINE_AA
                    font = cv2.FONT_HERSHEY_DUPLEX
                    scale = .4
                    (twl, thl) = cv2.getTextSize(label, font, fontScale=scale, thickness=1)[0]                    
                    oxl = last_pos[2]
                    oyl = last_pos[3] - 5                    
                    cv2.rectangle(frame, (last_pos[2], last_pos[3]), (last_pos[4], last_pos[5]), color, 1)
                    cv2.rectangle(frame, (oxl, oyl + 4), (oxl + twl + 4, oyl - thl - 1), color, cv2.FILLED)
                    cv2.putText(frame, label, (oxl + 2, oyl), font, scale, white, thickness=1, lineType=line)
                    success, speed = obj.measure_speed(self.framerate)
                    if success:
                        # Only measure the speed if we have 3 frames of information
                        text = "{:.3f} px/s".format(speed)
                        (tws, ths) = cv2.getTextSize(text, font, fontScale=scale, thickness=1)[0]
                        oxs = last_pos[2]
                        oys = last_pos[5] - 5
                        cv2.rectangle(frame, (oxs, oys + 4), (oxs + tws + 4, oys - ths - 1), color, cv2.FILLED)
                        cv2.putText(frame, text, (oxs + 2, oys + 2), font, scale, white, thickness=1, lineType=line)                    

        # Unregister the objects that haven't been detected for two consecutive frames
        to_unregister = set(self.objects.keys()) - set(matches.keys())
        for obj_id in to_unregister:
            if self.objects[obj_id].get_missing_count() > 2:
                del self.objects[obj_id]
            else:
                self.objects[obj_id].increment_missing_count()

        if self.debug:
            # Save the frame before the speed label is drawn
            TrackingDebugger.save_frame(frame, frame_id, self.classes, predictions, self.startup_time)

        return self.objects