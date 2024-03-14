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

def local_server():
    host = params.local_server
    port = params.local_port

    local_socket = socket.socket() 
    local_socket.bind((host, port))

    detector = ObjectDetector()
    tracker = ObjectTracker()
    history = ObstacleLocationHistory()
    controller = Controller()
    planner = WaypointPlanner(None)

    print("Local server open...")

    # configure how many client the server can listen simultaneously
    local_socket.listen(10)
    local_conn, address = local_socket.accept()  

    if params.perception_loc == 'cloud' or params.control_loc == 'cloud':
        host = params.cloud_server
        port = params.cloud_port
        cloud_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        cloud_conn.connect((host, port))
        print("Connected to cloud..")

    while True:
        # receive data stream. it won't accept data packet greater than 1024 bytes
        start_rx = time.time()
        input = recv_msg(local_conn)
        print("Sensor message rx time: "+str(time.time()-start_rx))
        # input_message = pickle.loads(input)

        if params.perception_loc == 'cloud':
            start_tx = time.time()
            send_msg(cloud_conn, input)
            print("Perception message tx time to cloud: "+str(time.time()-start_tx))
            
            if params.control_loc == 'cloud':
                start_rx = time.time()
                input = recv_msg(cloud_conn)
                print("Control message rx time from cloud: "+str(time.time()-start_rx))
                input_message = pickle.loads(input)
                control_msg = ControlMessage(steer=input_message.steer, throttle=input_message.throttle, brake=input_message.brake, hand_brake=False, reverse=False, timestamp=0)
                send_msg(local_conn, pickle.dumps(control_msg))
            
            elif params.control_loc == 'local':
                start_rx = time.time()
                input = recv_msg(cloud_conn)
                print("Planner message rx time from cloud: "+str(time.time()-start_rx))
                input_message = pickle.loads(input)
                (steer, throttle, brake, controller_runtime) = controller.get_control_instructions(timestamp, input_message.pose, input_message.waypoints)
                print("Control instructions {} {} {} {}".format(throttle, steer, brake, controller_runtime))
                control_msg = ControlMessage(steer=steer, throttle=throttle, brake=brake, hand_brake=False, reverse=False, timestamp=0)
                start_tx = time.time()
                send_msg(local_conn, pickle.dumps(control_msg))
                print("Control message tx time to sim: "+str(time.time()-start_tx))
                print("Sent control message")
        
        elif params.perception_loc == 'local':
            input_message = pickle.loads(input)
            timestamp = input_message.timestamp
            sensor_data = SensorMessage(timestamp = input_message.timestamp,
                                        frame=input_message.frame,
                                        depth_frame=input_message.depth_frame,
                                        pose=input_message.pose)

            obstacles = []
            tracked_obstacles = []
            obstacle_trajectories = []
            obstacle_predictions = []
            waypoints = None

            (timestamp, obstacles, detector_runtime) = detector.get_obstacles(timestamp, sensor_data.frame)
            print("Detected obstacles {} {}".format(len(obstacles), detector_runtime))
            
            (timestamp, tracked_obstacles, tracker_runtime) = tracker.get_tracked_obstacles(timestamp, sensor_data.frame, obstacles)
            print("Tracked obstacles  {} {}".format(len(obstacles), tracker_runtime))
            
            if len(tracked_obstacles) > 0:
                print(obstacles)
                print(tracked_obstacles)
                (timestamp, obstacle_trajectories) = history.get_location_history(timestamp, sensor_data.pose, sensor_data.depth_frame, tracked_obstacles)
                print("Trajectories       {} ".format(len(obstacle_trajectories)))
            
            if len(obstacle_trajectories) > 0:
                # first_trajectory = obstacle_trajectories[0].trajectory
                # for traj_location in first_trajectory:
                #     print("Trajectory 1  - " + str(traj_location))
                obstacle_trajectories_message = ObstacleTrajectoriesMessage(obstacle_trajectories) # necessary because this contains methods used in prediction
                (obstacle_predictions, predictor_runtime) = get_predictions(obstacle_trajectories_message)
                print("Predictions        {} {}".format(len(obstacle_predictions), predictor_runtime))
            
            if len(obstacle_predictions) > 0:
                print("Predictions  - " + str(obstacle_predictions[0]))
                (waypoints, planner_runtime) = planner.get_waypoints(timestamp, sensor_data.pose, obstacle_predictions)
                print("Planner waypoints  {} {}".format(len(waypoints.waypoints), planner_runtime))
            
            if params.control_loc == 'local':
                (steer, throttle, brake, controller_runtime) = controller.get_control_instructions(timestamp, sensor_data.pose, waypoints)
                print("Control instructions {} {} {} {}".format(throttle, steer, brake, controller_runtime))
                control_msg = ControlMessage(steer=steer, throttle=throttle, brake=brake, hand_brake=False, reverse=False, timestamp=0)
                cmd = pickle.dumps(control_msg)
                start_tx = time.time()
                print("----------------------Sent control data"+str(len(cmd)))
                send_msg(local_conn, cmd)

            elif params.control_loc == 'cloud':
                planner_msg = PlannerMessage(pose=sensor_data.pose, waypoints=waypoints)
                pmd = pickle.dumps(planner_msg)
                print("----------------------Sent planner data"+str(len(pmd)))
                start_tx = time.time()
                send_msg(cloud_conn, pmd)
                print("Planner message tx time to cloud: "+str(time.time()-start_tx))
                start_rx = time.time()
                input = recv_msg(cloud_conn)
                print("Control message rx time from cloud: "+str(time.time()-start_rx))
                input_message = pickle.loads(input)
                control_msg = ControlMessage(steer=input_message.steer, throttle=input_message.throttle, brake=input_message.brake, hand_brake=False, reverse=False, timestamp=timestamp)
                cmd = pickle.dumps(control_msg)
                print("----------------------Sent control data"+str(len(cmd)))
                start_tx = time.time()
                send_msg(local_conn, cmd)
                print("Control message tx time to sim: "+str(time.time()-start_tx))


if __name__=='__main__':
    #thread_one = threading.Thread(target=controller_server)
    #thread_two = threading.Thread(target=tracker_server)
    #thread_one.start()
    #thread_two.start()
    local_server()
