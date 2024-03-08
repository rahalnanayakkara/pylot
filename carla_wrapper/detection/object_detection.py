import params
import time
import numpy as np
import tensorflow as tf
import utils.logging

from objects.objects import BoundingBox2D, Obstacle
from objects.messages import ObstaclesMessage

from utils.detection_utils import OBSTACLE_LABELS
from utils.detection_utils import load_coco_labels, load_coco_bbox_colors

obstacle_detection_gpu_index = params.obstacle_detection_gpu_index
obstacle_detection_model_paths = params.obstacle_detection_model_paths
path_coco_labels = params.path_coco_labels
obstacle_detection_min_score_threshold = params.obstacle_detection_min_score_threshold

class ObjectDetector:

    def __init__(self):
        self._config_name = "ObjectDetector"
        self._module_logger = utils.logging.get_module_logger(self._config_name)
        self._csv_logger = utils.logging.ModuleCompletionLogger()

        print("\nInitializing Object Detector ... ")
        self._module_logger.info("\nInitializing Object Detector")
        
        # Only sets memory growth for flagged GPU
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices([physical_devices[obstacle_detection_gpu_index]], 'GPU')
        tf.config.experimental.set_memory_growth(physical_devices[obstacle_detection_gpu_index], True)

        # Load the model from the saved_model format file.
        self._model = tf.saved_model.load(obstacle_detection_model_paths)

        self._coco_labels = load_coco_labels(path_coco_labels)
        self._bbox_colors = load_coco_bbox_colors(self._coco_labels)

        # Unique bounding box id. Incremented for each bounding box.
        self._unique_id = 0

        # Serve some junk image to load up the model.
        self.__run_model(np.zeros((108, 192, 3), dtype='uint8'))


    def get_obstacles(self, timestamp, frame):
        self._module_logger.info('@{}: {} getting obstacles'.format(timestamp, self._config_name))
        start_time = time.time()
        # The models expect BGR images.
        assert frame.encoding == 'BGR', 'Expects BGR frames'
        num_detections, res_boxes, res_scores, res_classes = self.__run_model(frame.frame)
        obstacles = []
        for i in range(0, num_detections):
            if res_classes[i] in self._coco_labels:
                if (res_scores[i] >=
                         obstacle_detection_min_score_threshold):
                    if (self._coco_labels[res_classes[i]] in OBSTACLE_LABELS):
                        obstacles.append(
                            Obstacle(BoundingBox2D(
                                int(res_boxes[i][1] * frame.camera_setup.width),
                                int(res_boxes[i][3] * frame.camera_setup.width),
                                int(res_boxes[i][0] * frame.camera_setup.height),
                                int(res_boxes[i][2] * frame.camera_setup.height)),
                                     res_scores[i],
                                     self._coco_labels[res_classes[i]],
                                     id=self._unique_id)
                                )
                        self._unique_id += 1
            else:
                self._module_logger.warning('Filtering unknown class: {}'.format(res_classes[i]))

        self._module_logger.info('@{}: {} obstacles: {}'.format(timestamp, self._config_name, obstacles))

        # Get runtime in ms.
        runtime = (time.time() - start_time) * 1000
        self._csv_logger.log_module_completion(self._config_name, runtime)
        # Send out obstacles.
        return (timestamp, obstacles, runtime)


    def __run_model(self, image_np):
        # Expand dimensions since the model expects images to have
        # shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        infer = self._model.signatures['serving_default']
        result = infer(tf.convert_to_tensor(value=image_np_expanded))

        boxes = result['boxes']
        scores = result['scores']
        classes = result['classes']
        num_detections = result['detections']

        num_detections = int(num_detections[0])
        res_classes = [int(cls) for cls in classes[0][:num_detections]]
        res_boxes = boxes[0][:num_detections]
        res_scores = scores[0][:num_detections]
        return num_detections, res_boxes, res_scores, res_classes

        


