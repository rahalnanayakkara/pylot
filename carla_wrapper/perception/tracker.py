from sort.sort import Sort

import time
import numpy as np

from objects.objects import BoundingBox2D, Obstacle
from objects.messages import ObstaclesMessage

import params

def get_tracker_output_message(frame, obstacles, reinit, tracker):  
    start = time.time()
    
    detected_obstacles = []
    tracked_obstacles = []
    for obstacle in obstacles:
        if obstacle.is_vehicle() or obstacle.is_person():
            detected_obstacles.append(obstacle)
    tracker.reinitialize(frame, detected_obstacles)

    _, tracked_obstacles = tracker.track(frame)
    runtime = 1000 * (time.time() - start)
    return ObstaclesMessage(tracked_obstacles, runtime)


class MultiObjectTracker(object):
    def __init__(self):
        self._trackers = []

    def reinitialize(self, frame, obstacles):
        """ Reinitializes a multiple obstacle tracker.

        Args:
            frame: frames.CameraFrame to reinitialize with.
            obstacles: List of objects.Obstacle.
        """
        pass

    def track(self, frame):
        """ Tracks obstacles in a frame.

        Args:
            frame: frames.CameraFrame to track in.
        """
        return True, [tracker.track(frame) for tracker in self._trackers]


class MultiObjectSORTTracker(MultiObjectTracker):
    def __init__(self):
        self.tracker = Sort(max_age=params.obstacle_track_max_age,
                            min_hits=1,
                            min_iou=params.min_matching_iou)

    def reinitialize(self, frame, obstacles):
        """ Reinitializes a multiple obstacle tracker.

        Args:
            frame (:py:class:`~frames.CameraFrame`):
                Frame to reinitialize with.
            obstacles : List of objects.Obstacle.
        """
        detections, labels, ids = self.convert_detections_for_sort_alg(
            obstacles)
        self.tracker.update(detections, labels, ids)

    def track(self, frame):
        """ Tracks obstacles in a frame.

        Args:
            frame (:py:class:`~frames.CameraFrame`):
                Frame to track in.
        """
        # each track in tracks has format ([xmin, ymin, xmax, ymax], id)
        obstacles = []
        for track in self.tracker.trackers:
            coords = track.predict()[0].tolist()
            # changing to xmin, xmax, ymin, ymax format
            xmin = int(coords[0])
            xmax = int(coords[2])
            ymin = int(coords[1])
            ymax = int(coords[3])
            if xmin < xmax and ymin < ymax:
                bbox = BoundingBox2D(xmin, xmax, ymin, ymax)
                obstacles.append(Obstacle(bbox, 0, track.label, track.id))
            else:
                print(
                    "Tracker found invalid bounding box {} {} {} {}".format(
                        xmin, xmax, ymin, ymax))
        return True, obstacles

    def convert_detections_for_sort_alg(self, obstacles):
        converted_detections = []
        labels = []
        ids = []
        for obstacle in obstacles:
            bbox = [
                obstacle.bounding_box_2D.x_min, obstacle.bounding_box_2D.y_min,
                obstacle.bounding_box_2D.x_max, obstacle.bounding_box_2D.y_max,
                obstacle.confidence
            ]
            converted_detections.append(bbox)
            labels.append(obstacle.label)
            ids.append(obstacle.id)
        return (np.array(converted_detections), labels, ids)


class MultiObjectDeepSORTTracker(MultiObjectTracker):
    def __init__(self):
        # self._logger = logger
        # Initialize the deepsort object, which has a tracker object within it
        from dependencies.nanonets_object_tracking.deepsort import deepsort_rbc
        self._deepsort = deepsort_rbc(
            wt_path=params.deep_sort_tracker_weights_path,
            max_age=params.obstacle_track_max_age,
            min_iou=params.min_matching_iou)

    def reinitialize(self, frame, obstacles):
        """ Reinitializes a multiple obstacle tracker.

        Args:
            frame (:py:class:`~frames.CameraFrame`):
                Frame to reinitialize with.
            obstacles: List of objects.Obstacle.
        """
        # update tracker with new detections
        self.track(frame, obstacles)

    def track(self, frame, obstacles=[]):
        """ Tracks obstacles in a frame.

        Args:
            frame (:py:class:`~frames.CameraFrame`):
                Frame to track in.
        """
        # If obstacles, run deep sort to update tracker with detections.
        # Otherwise, step each confirmed track one step forward.
        if obstacles:
            bboxes, labels, confidence_scores, ids = [], [], [], []
            for obstacle in obstacles:
                bboxes.append(obstacle.bounding_box_2D.as_width_height_bbox())
                labels.append(obstacle.label)
                confidence_scores.append(obstacle.confidence)
                ids.append(obstacle.id)
            self._deepsort.run_deep_sort(frame.frame, confidence_scores,
                                         bboxes, labels, ids)
        else:
            for track in self._deepsort.tracker.tracks:
                if track.is_confirmed():
                    track.predict(self._deepsort.tracker.kf)
        tracked_obstacles = []
        for track in self._deepsort.tracker.tracks:
            if track.is_confirmed():
                # Converts x, y, w, h bbox to tlbr bbox (top left and bottom
                # right coords).
                bbox = track.to_tlbr()
                # Converts to xmin, xmax, ymin, ymax format.
                xmin = int(bbox[0])
                xmax = int(bbox[2])
                ymin = int(bbox[1])
                ymax = int(bbox[3])
                if xmin < xmax and ymin < ymax:
                    bbox = BoundingBox2D(xmin, xmax, ymin, ymax)
                    tracked_obstacles.append(
                        Obstacle(bbox, 0, track.label, track.track_id))
                else:
                    print(
                        "Tracker found invalid bounding box {} {} {} {}".
                        format(xmin, xmax, ymin, ymax))
        return True, tracked_obstacles
