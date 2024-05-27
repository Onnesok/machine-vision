from detection import *
#All model zoo is here ==>  https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d7_coco17_tpu-32.tar.gz"
#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz"
#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz"
#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v1_fpn_640x640_coco17_tpu-8.tar.gz"

class_file = 'coco.names'
image_path = 'im1.png'
treshold = 0.5

detection = detection()
detection.readclasses(class_file)
detection.downnload_model(modelURL)
detection.loadModel()
#detection.predict_image(image_path, treshold)
detection.predict_camera(treshold)