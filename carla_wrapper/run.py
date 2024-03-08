import time
import params

from utils.logging import setup_pipeline_logging, ModuleCompletionLogger
from simulation import CarlaSimulation
from detection.object_detection import ObjectDetector
from perception.object_tracking import ObjectTracker
from perception.location_history import ObstacleLocationHistory
from objects.messages import ObstacleTrajectoriesMessage
from prediction.predictor import get_predictions
from planning.planner import WaypointPlanner
from control.controller import Controller
from visualizer import Visualizer

class SimulationRunner():

    def __init__(self):
        self._simulation = CarlaSimulation()
        self._detector = ObjectDetector()
        self._tracker = ObjectTracker()
        self._history = ObstacleLocationHistory()
        # self._predictor = ...
        self._planner = WaypointPlanner()
        self._controller = Controller()
        self._visualizer = Visualizer()

    def run_one_tick(self):
        (frame, depth_frame, pose, timestamp)            = self._simulation.tick_simulator()
        (timestamp, obstacles, detector_runtime)         = self._detector.get_obstacles(timestamp, frame)
        (timestamp, tracked_obstacles, tracker_runtime)  = self._tracker.get_tracked_obstacles(timestamp, frame, obstacles)
        (timestamp, obstacle_trajectories)               = self._history.get_location_history(timestamp, pose, depth_frame, tracked_obstacles)
        
        obstacle_trajectories_message = ObstacleTrajectoriesMessage(timestamp, obstacle_trajectories) # necessary because this contains methods used in prediction

        (obstacle_predictions, predictor_runtime)    = get_predictions(obstacle_trajectories_message)
        (waypoints, planner_runtime)                 = self._planner.get_waypoints(obstacle_predictions)
        (throttle, steer, brake, controller_runtime) = self._controller.get_control_instructions(pose, waypoints)

        self._simulation.apply_control(throttle, steer, brake, False, False)
        self._visualizer.visualize(timestamp, frame, depth_frame, pose, obstacles, throttle, steer, brake)
        
        print("\nRuntime: {}\t{}\t{}\t{}\t{}".format(detector_runtime, tracker_runtime, predictor_runtime, planner_runtime, controller_runtime))
        print("\nLocation: {}, Control: {} {} {}", pose.transform.location, throttle, steer, brake)
    

def main():
    #setup_pipeline_logging()
    runner = SimulationRunner()
    while True:
        time.sleep(2)
        runner.run_one_tick()

if __name__ == '__main__':
    main()