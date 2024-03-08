import time
import params

from utils.logging import setup_pipeline_logging, ModuleCompletionLogger
from utils.simulation import get_world

from simulation import CarlaSimulation
from visualizer import Visualizer

class SimulationRunner():

    def __init__(self):

        from detection.object_detection import ObjectDetector
        from perception.object_tracking import ObjectTracker
        from perception.location_history import ObstacleLocationHistory
        from planning.planner import WaypointPlanner
        from control.controller import Controller

        client, world = get_world(params.simulator_host,
                             params.simulator_port,
                             params.simulator_timeout)

        self._simulation = CarlaSimulation(client, world)
        self._detector = ObjectDetector()
        self._tracker = ObjectTracker()
        self._history = ObstacleLocationHistory()
        # self._predictor = ...
        self._planner = WaypointPlanner()
        self._controller = Controller()
        self._visualizer = Visualizer(world)

    def run_one_tick(self):

        from objects.messages import ObstacleTrajectoriesMessage
        from prediction.predictor import get_predictions
        
        (timestamp, frame, depth_frame, pose)            = self._simulation.tick_simulator()
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
    
class MockSimulationRunner():
    
    def __init__(self):
        client, world = get_world(params.simulator_host,
                             params.simulator_port,
                             params.simulator_timeout)
        self._simulation = CarlaSimulation(client, world)
        self._visualizer = Visualizer(world)
    
    def run_one_tick(self):
        (timestamp, frame, depth_frame, pose) = self._simulation.tick_simulator()
        self._simulation.apply_control(1, 0, 0, False, False)
        self._visualizer.visualize(timestamp, frame, depth_frame, pose, None, 1, 0, 0)

def main():
    #setup_pipeline_logging()
    runner = MockSimulationRunner()
    while True:
        time.sleep(2)
        runner.run_one_tick()

if __name__ == '__main__':
    main()