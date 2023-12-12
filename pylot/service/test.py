from collections import deque

from objects import Waypoints, Obstacle, BoundingBox2D
from utils import Pose, Transform, Vector3D, Location, Rotation
from messages import ControlMessage, ObstaclesMessage, WaypointsMessage

from service import ControllerInput

from controller import get_control_message
from tracker import get_obstacle_tracker_message

import tracker
import socket
import pickle

def create_transform(x=0, y=0, z=1):
    return Transform(location=Location(x, y, z), rotation=Rotation(0, 0.5, 0))

def create_bounding_box(xmin, xmax, ymin, ymax):
    return BoundingBox2D(xmin, xmax, ymin, ymax)

def create_obstacle(x, y, z, xmin, xmax, ymin, ymax):
    transform = create_transform(x, y, z)
    bb = create_bounding_box(xmin, xmax, ymin, ymax)
    return Obstacle(bb, 1.0, "label", 1, transform, "dlabel", bb)


# Pass in a mocked pose + waypoint messages to controller
# Test PID, MPC
def test_controller():
    pose = Pose(
        transform=create_transform(0, 0, 1),
        forward_speed=1.0,
        velocity_vector=Vector3D(1, 1, 0),
        localization_time=0.0)

    waypoints=Waypoints(
        waypoints=deque([
            create_transform(5, 5, 1),
            create_transform(5, 10, 1),
            create_transform(10, 20, 1),
            create_transform(20, 20, 1),
            create_transform(20, 30, 1),
            create_transform(30, 30, 1),
            create_transform(30, 40, 1),
            create_transform(40, 40, 1),
            create_transform(40, 50, 1),
            create_transform(50, 60, 1)]),
        target_speeds=deque([5, 7, 6, 4, 6, 4, 5, 6, 5, 7])
    )
    
    print(get_control_message(pose=pose, waypoints=waypoints, type="pid"))
    print(get_control_message(pose=pose, waypoints=waypoints, type="mpc"))

    return 0

# Pass in a mocked obstacle message to controller
# Test sort, deep_sort
def test_tracker():
    
    obstacles=[
        create_obstacle(5, 5, 1, 20, 22, 10, 15),
        create_obstacle(5, 7, 1, 18, 21, 12, 15),
        create_obstacle(8, 7, 1, 15, 20, 15, 20)
    ]

    sort_tracker = tracker.MultiObjectSORTTracker()
    #deepsort_tracker = tracker.MultiObjectDeepSORTTracker()    

    print(get_obstacle_tracker_message(frame=None, obstacles=obstacles, reinit=True, tracker=sort_tracker))
    #print(get_obstacle_tracker_message(frame=None, obstacles=obstacles, reinit=True, tracker=deepsort_tracker))
    return ""

def test_remote_controller():
    host = "0.0.0.0"
    port = 5010  # initiate port no above 1024
        
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))

    pose = Pose(
        transform=create_transform(0, 0, 1),
        forward_speed=1.0,
        velocity_vector=Vector3D(1, 1, 0),
        localization_time=0.0)

    waypoints=Waypoints(
        waypoints=deque([
            create_transform(5, 5, 1),
            create_transform(5, 10, 1),
            create_transform(10, 20, 1),
            create_transform(20, 20, 1),
            create_transform(20, 30, 1),
            create_transform(30, 30, 1),
            create_transform(30, 40, 1),
            create_transform(40, 40, 1),
            create_transform(40, 50, 1),
            create_transform(50, 60, 1)]),
        target_speeds=deque([5, 7, 6, 4, 6, 4, 5, 6, 5, 7])
    )
        
    controller_input = ControllerInput(pose_msg=pose, waypoints_msg=waypoints, type="pid")
    input_string = pickle.dumps(controller_input)

    s.send(input_string)
    output_string = s.recv(102400)

    control_msg = pickle.loads(output_string)
    print("Received control message ", control_msg)


if __name__=='__main__':
    test_controller()
    # test_remote_controller()
    test_tracker()
