import socket
import pickle
import threading
import time
import params
import numpy as np

from detection.object_detection import ObjectDetector
from perception.object_tracking import ObjectTracker
from perception.location_history import ObstacleLocationHistory
from objects.messages import ObstacleTrajectoriesMessage
from planning.planner import WaypointPlanner
from control.controller import Controller

from objects.messages import SensorMessage, ControlMessage, PlannerMessage
from prediction.predictor import get_predictions
from utils.service import send_msg, recv_msg
from objects.frames import CameraFrame, DepthFrame
from objects.objects import CameraSetup, DepthCameraSetup, Transform, Location, Rotation, Pose, Vector3D

def server():
    print("In server")

    detector = ObjectDetector()
    print("detector")
    tracker = ObjectTracker()
    print("tracker")
    history = ObstacleLocationHistory()
    print("olh")
    controller = Controller()
    print("controller")
    planner = WaypointPlanner(None)
    print("planner")

    frame_temp = np.zeros((512, 960, 3), dtype='uint8')
    frame = frame_temp.astype(np.float32)
    frame = np.dot(frame[:, :, :3], [65536.0, 256.0, 1.0])
    frame /= 16777215.0

    while True:

        if params.perception_loc == 'local':

            print("In main function")

            sensor_data = SensorMessage(timestamp = 8357,
                                        frame=CameraFrame(encoding= 'BGR', 
                                                            frame = np.zeros((512, 960, 3), dtype='uint8'),
                                                            camera_setup= CameraSetup(name= 'center_camera', 
                                                            camera_type= 'sensor.camera.rgb', 
                                                            width= 960, 
                                                            height= 512, 
                                                            transform= Transform(location= Location(x=1.3, y=0.0, z=1.8), rotation= Rotation(pitch=-15, yaw=0, roll=0)), 
                                                            fov= 90.0)),
                                        depth_frame=DepthFrame(frame = frame,
                                                            camera_setup= CameraSetup(name= 'depth_center_camera', 
                                                            camera_type= 'sensor.camera.depth', 
                                                            width= 960, 
                                                            height= 512, 
                                                            transform= Transform(location= Location(x=1.3, y=0.0, z=1.8), rotation= Rotation(pitch=-15, yaw=0, roll=0)), 
                                                            fov= 90.0)),
                                        pose=Pose(transform= Transform(location= Location(x=396.0, y=275.0, z=0.032674599438905716), rotation= Rotation(pitch=0.0, yaw=-89.99999237060547, roll=0.0)),
                                                            forward_speed= 0.0, 
                                                            velocity_vector= Vector3D(x=0.0, y=0.0, z=0.0)))

            obstacles = []
            tracked_obstacles = []
            obstacle_trajectories = []
            obstacle_predictions = []
            waypoints = None

            timestamp = sensor_data.timestamp

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
                exit()
                send_msg(cloud_conn, pickle.dumps(control_msg))
                print("Sent control message")

        #     elif params.control_loc == 'local':
        #         planner_msg = PlannerMessage(pose=sensor_data.pose, waypoints=waypoints)
        #         send_msg(cloud_conn, pickle.dumps(planner_msg))
        #         print("Sent planner message")

        # elif params.perception_loc == 'local' and params.control_loc == 'cloud':
        #     input = recv_msg(cloud_conn)
        #     input_message = pickle.loads(input)
        #     (steer, throttle, brake, controller_runtime) = controller.get_control_instructions(input_message.pose, input_message.waypoints)
        #     print("Control instructions {} {} {} {}".format(throttle, steer, brake, controller_runtime))
        #     control_msg = ControlMessage(steer=steer, throttle=throttle, brake=brake, hand_brake=False, reverse=False, timestamp=0)
        #     send_msg(cloud_conn, pickle.dumps(control_msg))
        #     print("Sent control message")

    #     print("Generated tracker message: ", tracker_message)
    #     conn.send(pickle.dumps(tracker_message))  # send data to the client
    # conn.close()  # close the connection


if __name__=='__main__':
    #thread_one = threading.Thread(target=controller_server)
    #thread_two = threading.Thread(target=tracker_server)
    #thread_one.start()
    #thread_two.start()
    server()
