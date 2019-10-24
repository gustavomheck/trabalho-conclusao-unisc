import math

class TrackableObject:
    def __init__(self, object_id, color, detection):
        self.object_id = object_id
        self.positions = [detection]
        self.color = color
        self.missing_count = 0

    def measure_speed(self, framerate):
        pos_len = len(self.positions)
        if pos_len < 3:
            return False, None

        (a, b) = self.positions[-3][1]
        (c, d) = self.positions[-1][1]
        dist = math.sqrt(pow(a - c, 2) + pow(b - d, 2))
        time = (framerate * 7.0)
        speed = dist / time
        return True, speed

    def get_last_position(self):
        return self.positions[-1]

    def get_id(self):
        return self.object_id

    def get_color(self):
        return self.color        
    
    def get_missing_count(self):
        return self.missing_count

    def increment_missing_count(self):
        self.missing_count += 1
    
    def reset_missing_count(self):
        self.missing_count = 0

    def append_position(self, obj):
        self.positions.append(obj)