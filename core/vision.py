import cv2
import os
import sys
from ultralytics import YOLO

class VisionProcessor:
    def __init__(self):
        # Detect if running as a PyInstaller bundle
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            # The weights file will be extracted here at runtime
            weights_path = os.path.join(sys._MEIPASS, 'yolov8n.pt')
        else:
            # Fallback for standard Python script execution
            weights_path = 'yolov8n.pt'
            
        self.model = YOLO(weights_path)

    def detect_objects(self, image_path):
        results = self.model(image_path)
        
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        bounding_boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width = x2 - x1
                height = y2 - y1
                conf = float(box.conf[0]) 
                bounding_boxes.append((x1, y1, width, height, conf))
                
        return img_rgb, bounding_boxes