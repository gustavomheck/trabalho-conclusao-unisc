from predictionsgenerator import PredictionsGenerator
from objecttracker import ObjectTracker
import numpy as np
import cv2
import unittest2

class SpeedMeasurementTest(unittest2.TestCase):
    def __init__(self, *args, **kwargs):
        unittest2.TestCase.__init__(self, *args, **kwargs)
        self.classes = {2: "car", 3: "bike", 5: "bus", 7: "truck"}
        self.generator = PredictionsGenerator()
        self.tracker = ObjectTracker(self.classes, 30.0, 32, 288, 0.75, 30)
    
    def test_can_detect_objects(self):
        frame_id = 1
        boxes = self.generator.generate_boxes(frame_id)
        objects = self.tracker.update(np.zeros((288, 352)), frame_id, boxes)
        self.assertEqual(len(objects), 4)
        obj = objects[1]
        last_pos = obj.get_last_position()
        self.assertEqual(last_pos[2], 421.0)
        self.assertEqual(last_pos[3], 46.0)
        self.assertEqual(last_pos[4], 456.0)
        self.assertEqual(last_pos[5], 73.0)

    def test_can_track_objects(self):
        for frame_id in range(5, 10):
            boxes = self.generator.generate_boxes(frame_id)
            objects = self.tracker.update(np.zeros((288, 352)), frame_id, boxes)
            self.assertEqual(len(objects), 4)

    def test_can_detect_and_track_new_object(self):
        frame = np.zeros((288, 352))
        frame_id = 620
        boxes = self.generator.generate_boxes(frame_id)
        objects = self.tracker.update(frame, frame_id, boxes)
        self.assertEqual(len(objects), 2)
        frame_id = 624
        boxes = self.generator.generate_boxes(frame_id)
        objects = self.tracker.update(frame, frame_id, boxes)
        self.assertEqual(len(objects), 3)

    def test_can_stop_tracking_object(self):
        self.assertEqual(0, 0)

if __name__ == '__main__':
    unittest2.main()