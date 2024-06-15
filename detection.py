import cv2, time, os, tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(100)

class detection:
    def __init__(self):
        self.class_list = []
        self.color_list = []
        self.model = None
    
    
    def readclasses(self, classes_file_path):
        with open(classes_file_path, 'r') as f:
            self.class_list = f.read().splitlines()
            
        #color list
        self.color_list = np.random.randint(0, 255, size=(len(self.class_list), 3), dtype=int)
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
        
        tf.compat.v1.reset_default_graph()
        self.model = tf.saved_model.load(model_path)    
        print("Model", self.model_name, "successfully loaded")


    def create_boundingbox(self, image, threshold=0.5):
        input_tensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.uint8)
        input_tensor = input_tensor[tf.newaxis, ...]
        
        detections = self.model(input_tensor)
        bbox = detections['detection_boxes'][0].numpy()
        class_index = detections['detection_classes'][0].numpy().astype(np.int32)
        class_scores = detections['detection_scores'][0].numpy()
        
        imH, imW, imC = image.shape
        bbox_index = tf.image.non_max_suppression(bbox, class_scores, max_output_size=50, iou_threshold=threshold, score_threshold=threshold)
        #print(bbox_index)
        
        for i in bbox_index:
            #print(bbox[i])
            ymin, xmin, ymax, xmax = bbox[i]
            class_confidence = 100 * class_scores[i]
            class_label_text = self.class_list[class_index[i]].upper()
            class_color = tuple(self.color_list[class_index[i]].tolist())
            
            display_text = f'{class_label_text}: {class_confidence:.2f}%'
            xmin, xmax, ymin, ymax = int(xmin * imW), int(xmax * imW), int(ymin * imH), int(ymax * imH)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), class_color, thickness=2)
            
            (text_width, text_height), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(image, (xmin, ymin - text_height - 10), (xmin + text_width, ymin), class_color, -1)
            cv2.putText(image, display_text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
        return image


            
    def predict_image(self, img_path, treshold):
        image = cv2.imread(img_path)
        
        bbox_image = self.create_boundingbox(image)
        
        cv2.imwrite(self.model_name + ".png", bbox_image)
        cv2.imshow("Result", bbox_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

    def predict_camera(self, video_url = 0, threshold=0.5):
        cap = cv2.VideoCapture(video_url)
        prev_time = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            #fps calc
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            
            
            flipped_frame = cv2.flip(frame, 1)
            bbox_frame = self.create_boundingbox(flipped_frame, threshold)
            
            fps_text = f"FPS: {int(fps)}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2
            text_size, _ = cv2.getTextSize(fps_text, font, font_scale, font_thickness)
            text_w, text_h = text_size
            text_x = 10
            text_y = 30
            
            fps_back = cv2.rectangle(bbox_frame, (text_x, text_y - text_h - 10), (text_x + text_w + 10, text_y + 10), (100, 100, 100), -1)
            cv2.putText(fps_back, fps_text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)
            
            # Create named window
            cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
            
            # Set window size
            cv2.resizeWindow("Frame", 800, 600)
            
            cv2.imshow('Frame', bbox_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        
        
