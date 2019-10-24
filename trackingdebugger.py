import cv2
import os

class TrackingDebugger:
    @staticmethod
    def save_frame(frame, frame_id, classes,  predictions, time):
        output_path = "output\\" + time
        if not os.path.exists(output_path):
            os.mkdir(os.getcwd() + "\\" + output_path)
            
        output_path = output_path +  "\\"
        cv2.imwrite(output_path + str(frame_id) + ".jpg", frame)      

        with open(output_path + "boxes.txt", "a+") as f:
            comma = ","
            for obj_class in classes:
                preds_len = len(predictions[obj_class])
                if preds_len > 0:
                    f.write("Frame ID: {} Class: {}\n".format(frame_id, classes[obj_class]))
                    for i in range(0, preds_len):
                        (x1, y1, x2, y2, c) = predictions[obj_class][i]

                        if i == len(predictions[obj_class]) - 1:
                            comma = ""

                        f.write("[{}, {}, {}, {}, {}]{}\n".format(x1, y1, x2, y2, c, comma))
                    
                    f.write("\n")