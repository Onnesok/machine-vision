import cv2, time, os, tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file


# import cv2
# import time
# import os
# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.utils import get_file  # Updated import for get_file

np.random.seed(20)

class detection:
    def __init__(self):
        pass
    
    def readclasses(self, classes_file_path):
        with open(classes_file_path, 'r') as f:
            self.class_list = f.read().splitlines()
            
        #color list
        self.color_list = np.random.uniform(low=0,  high=255, size=(len(self.class_list)))
        #print(len(self.class_list), len(self.color_list))
        
    def downnload_model(self, modelURL):
        file_name = os.path.basename(modelURL)
        self.model_name = file_name[:file_name.index('.')]
        # print(file_name)
        # print(self.model_name)
        self.cache_dir = "./pretrained_models"
        os.makedirs(self.cache_dir, exist_ok=True)
        get_file(fname= file_name, origin=modelURL, cache_dir= self.cache_dir, cache_subdir= "checkpoints", extract=True)
        
        
        
    def loadModel(self):
        print("Loading model:", self.model_name)
        model_path = os.path.join(self.cache_dir, "checkpoints", self.model_name, "saved_model")
        print("Model path:", model_path)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SavedModel file does not exist at: {model_path}")
        
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(model_path)    
        print("Model", self.model_name, "successfully loaded")

        
    def create_boundingbox(self, image, treshold = 0.5):
        input_tensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.uint8)
        input_tensor = input_tensor[tf.newaxis, ...]
        
        detections = self.model(input_tensor)
        bbox = detections['detection_boxes'][0].numpy()
        class_index = detections['detection_classes'][0].numpy().astype(np.int32)
        class_scores = detections['detection_scores'][0].numpy()
        
        imH, imW, imC = image.shape
        bbox_index = tf.image.non_max_suppression(bbox, class_scores, max_output_size=50, iou_threshold=treshold, score_threshold=treshold)
        print(bbox_index)
        
        if len(bbox) != 0:
            for i in bbox_index:
                bbox_item = bbox[i]
                if not isinstance(bbox_item, np.ndarray):
                    print("bbox is not a NumPy array:", bbox_item)
                    continue
                
                print("bbox:", bbox_item)  # Debugging output
                
                if len(bbox_item) != 4:
                    print("Unexpected bbox structure:", bbox_item)
                    continue
                
                ymin, xmin, ymax, xmax = bbox_item
                
                class_confidence = (100 * class_scores[i])
                class_label_text = self.class_list[class_index[i]].upper()
                class_color = self.color_list[class_index[i]]
                
                display_text = '{}: {}%'.format(class_label_text, class_confidence)
                xmin, xmax, ymin, ymax = int(xmin * imW), int(xmax * imW), int(ymin * imH), int(ymax * imH)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=class_color, thickness=1)
                cv2.putText(image, display_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, class_color, 2)
                
                line_width = min(int((xmax - xmin)*0.2), int((ymax - ymin) * 0.2))
                cv2.line(image, (xmin, ymin), (xmin+line_width, ymin), class_color, thickness=5)
                cv2.line(image, (xmin, ymin), (xmin, line_width+ymin), class_color, thickness=5)
            
        return image


            
    def predict_image(self, img_path, treshold):
        image = cv2.imread(img_path)
        
        bbox_image = self.create_boundingbox(image)
        
        cv2.imwrite(self.model_name + ".png", bbox_image)
        cv2.imshow("Result", bbox_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

    def predict_camera(self, threshold=0.5):
        cap = cv2.VideoCapture(0)
        prev_time = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            
            bbox_frame = self.create_boundingbox(frame, threshold)
            cv2.putText(bbox_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Frame', bbox_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
