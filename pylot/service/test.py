from collections import deque

from objects import Waypoints, Obstacle, BoundingBox2D
from utils import Pose, Transform, Vector3D, Location, Rotation
from messages import ControlMessage, ObstaclesMessage, WaypointsMessage

from controller import get_control_message
from tracker import get_obstacle_tracker_message

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
    pose_msg = Pose(
        transform=create_transform(0, 0, 1),
        forward_speed=1.0,
        velocity_vector=Vector3D(1, 1, 0),
        localization_time=0.0)

    waypoint_msg = WaypointsMessage(waypoints=Waypoints(
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
    ), agent_state=0)
    
    print(get_control_message(pose_msg=pose_msg, waypoint_msg=waypoint_msg, type="pid"))
    print(get_control_message(pose_msg=pose_msg, waypoint_msg=waypoint_msg, type="mpc"))

    return 0


# Pass in a mocked obstacle message to controller
# Test sort, deep_sort
def test_tracker():
    obstacle_msg = ObstaclesMessage(
        obstacles=[
            create_obstacle(5, 5, 1, 20, 10, 22, 15),
            create_obstacle(5, 7, 1, 18, 12, 21, 15),
            create_obstacle(8, 7, 1, 15, 15, 20, 20)
        ], runtime=10.0)
    
    print(get_obstacle_tracker_message(frame_msg=None, obstacles_msg=obstacles_msg, type="sort"))
    print(get_obstacle_tracker_message(frame_msg=None, obstacles_msg=obstacles_msg, type="sort"))
    return ""

if __name__=='__main__':
    test_controller()