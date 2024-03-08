import time
from collections import deque
import time

import utils.logging
import params

class ObjectTracker():
    def __init__(self):

        self._config_name = "ObjectTracker"
        self._module_logger = utils.logging.get_module_logger(self._config_name)
        self._csv_logger = utils.logging.ModuleCompletionLogger()

        print("\nInitializing Object Tracker")
        self._module_logger.info("\nInitializing Object Tracker")

        self._tracker_type = params.tracker_type
        # Absolute time when the last tracker run completed.
        self._last_tracker_run_completion_time = 0
        try:
            if params.tracker_type == 'deep_sort':
                from tracker import MultiObjectDeepSORTTracker
                self._tracker = MultiObjectDeepSORTTracker
            elif params.tracker_type == 'sort':
                from tracker import MultiObjectSORTTracker
                self._tracker = MultiObjectSORTTracker
            else:
                raise ValueError(
                    'Unexpected tracker type {}'.format(self._tracker_type))
        except ImportError as error:
            self._module_logger.exception('Error importing {}'.format(self._tracker_type))
            raise error

        self._obstacles_msgs = deque()
        self._frame_msgs = deque()
        self._detection_update_count = -1
        self._server=None

    def _reinit_tracker(self, camera_frame, detected_obstacles):
        start = time.time()
        result = self._tracker.reinitialize(camera_frame, detected_obstacles)
        return (time.time() - start) * 1000, result

    def _run_tracker(self, camera_frame):
        start = time.time()
        result = self._tracker.track(camera_frame)
        return (time.time() - start) * 1000, result

    def get_tracked_obstacles(self, timestamp, camera_frame, obstacles):
        self._module_logger.info('@{}: ObjectTracker received message'.format(timestamp))
        if timestamp.is_top:
            return
        tracked_obstacles = []
        reinit_runtime = 0
        start_time = time.time()

        if params.tracker_source == 'server':
            tracker_output = self.fetch_from_server(camera_frame, tracked_obstacles, True)
            total_tracker_time = 1000 * (time.time() - start_time)
            print("Total Tracker time: ", total_tracker_time)
            tracked_obstacles = tracker_output.obstacles
            tracker_runtime = tracker_output.runtime
        else:
            detected_obstacles = []
            for obstacle in obstacles:
                if obstacle.is_vehicle() or obstacle.is_person():
                    detected_obstacles.append(obstacle)
            reinit_runtime, _ = self._reinit_tracker(camera_frame, detected_obstacles)
            tracker_runtime, (ok, tracked_obstacles) = self._run_tracker(camera_frame)
        
        assert ok, 'Tracker failed at timestamp {}'.format(timestamp)
        tracker_runtime = tracker_runtime + reinit_runtime
        self._csv_logger.log_module_completion(self._config_name, tracker_runtime)
        return (timestamp, tracked_obstacles, tracker_runtime)