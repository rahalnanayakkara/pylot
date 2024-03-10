import socket
import pickle
import threading
import time
import params

from detection.object_detection import ObjectDetector
from perception.object_tracking import ObjectTracker
from perception.location_history import ObstacleLocationHistory
from objects.messages import ObstacleTrajectoriesMessage
from planning.planner import WaypointPlanner
from control.controller import Controller

from objects.messages import SensorMessage, ControlMessage, PlannerMessage
from prediction.predictor import get_predictions
from utils.service import send_msg, recv_msg

def cloud_server():
    host = params.cloud_server
    port = params.cloud_port

    cloud_socket = socket.socket() 
    cloud_socket.bind((host, port))

    print("Cloud server open...")

    # configure how many client the server can listen simultaneously
    cloud_socket.listen(10)
    cloud_conn, address = cloud_socket.accept()  

    detector = ObjectDetector()
    tracker = ObjectTracker()
    history = ObstacleLocationHistory()
    controller = Controller()
    planner = WaypointPlanner(None)

    while True:

        if params.perception_loc == 'cloud':
            input = recv_msg(cloud_conn)
            input_message = pickle.loads(input)
            sensor_data = SensorMessage(timestamp = input_message.timestamp,
                                        frame=input_message.frame,
                                        depth_frame=input_message.depth_frame,
                                        pose=input_message.pose)

            obstacles = []
            tracked_obstacles = []
            obstacle_trajectories = []
            obstacle_predictions = []
            waypoints = None

            timestamp = input_message.timestamp

            (timestamp, obstacles, detector_runtime) = detector.get_obstacles(timestamp, sensor_data.frame)
            print("Detected obstacles {} {}".format(len(obstacles), detector_runtime))
            
            (timestamp, tracked_obstacles, tracker_runtime) = tracker.get_tracked_obstacles(timestamp, sensor_data.frame, obstacles)
            print("Tracked obstacles  {} {}".format(len(obstacles), tracker_runtime))
            
            if len(tracked_obstacles) > 0:
                (timestamp, obstacle_trajectories) = history.get_location_history(timestamp, sensor_data.pose, sensor_data.depth_frame, tracked_obstacles)
                print("Trajectories       {} ".format(len(obstacle_trajectories)))
            
            if len(obstacle_trajectories) > 0:
                first_trajectory = obstacle_trajectories[0].trajectory
                # for traj_location in first_trajectory:
                #     print("Trajectory 1  - " + str(traj_location))
                obstacle_trajectories_message = ObstacleTrajectoriesMessage(obstacle_trajectories) # necessary because this contains methods used in prediction
                (obstacle_predictions, predictor_runtime) = get_predictions(obstacle_trajectories_message)
                print("Predictions        {} {}".format(len(obstacle_predictions), predictor_runtime))
            
            if len(obstacle_predictions) > 0:
                print("Predictions  - " + str(obstacle_predictions[0]))
                (waypoints, planner_runtime) = planner.get_waypoints(sensor_data.pose, obstacle_predictions)
                print("Planner waypoints  {} {}".format(len(waypoints.waypoints), planner_runtime))
            
            
            if params.control_loc == 'cloud':
                (steer, throttle, brake, controller_runtime) = controller.get_control_instructions(sensor_data.pose, waypoints)
                print("Control instructions {} {} {} {}".format(throttle, steer, brake, controller_runtime))
                control_msg = ControlMessage(steer=steer, throttle=throttle, brake=brake, hand_brake=False, reverse=False, timestamp=timestamp)
                send_msg(cloud_conn, pickle.dumps(control_msg))
                print("Sent control message")

            elif params.control_loc == 'local':
                planner_msg = PlannerMessage(pose=sensor_data.pose, waypoints=waypoints)
                send_msg(cloud_conn, pickle.dumps(planner_msg))
                print("Sent planner message")

        elif params.perception_loc == 'local' and params.control_loc == 'cloud':
            input = recv_msg(cloud_conn)
            input_message = pickle.loads(input)
            (steer, throttle, brake, controller_runtime) = controller.get_control_instructions(input_message.pose, input_message.waypoints)
            print("Control instructions {} {} {} {}".format(throttle, steer, brake, controller_runtime))
            control_msg = ControlMessage(steer=steer, throttle=throttle, brake=brake, hand_brake=False, reverse=False, timestamp=0)
            send_msg(cloud_conn, pickle.dumps(control_msg))
            print("Sent control message")

    #     print("Generated tracker message: ", tracker_message)
    #     conn.send(pickle.dumps(tracker_message))  # send data to the client
    # conn.close()  # close the connection


if __name__=='__main__':
    #thread_one = threading.Thread(target=controller_server)
    #thread_two = threading.Thread(target=tracker_server)
    #thread_one.start()
    #thread_two.start()
    cloud_server()
