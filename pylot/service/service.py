'''
Standalone service to provide hooks into individual controllers and trackers
    Controllers - PID, MPC
    Trackers - sort, deep_sort, da_siam_rpn

Needs access to following independent components -

- Atomic utils!
    - pylot.utils
    - pylot.control.utils
    - pylot.control.mpc.utils
    - pylot.perception.detection.utils

- ERDOS Messages
    - pylot.perception.messages.ObstacleMessage
    - pylot.control.messages.ControlMessage
    - pylot.utils.Pose
    - pylot.planning.waypoints.Waypoints
    - pylot.planning.messages.WaypointsMessage
    - pylot.perception.messages.DepthFrameMessage
    - pylot.perception.messages.PointCloudMessage

- Control
    - pylot.control.pid.PIDLongitudinalController
    - pylot.control.mpc.mpc.ModelPredictiveController

- Trackers
    - pylot.perception.detection.obstacle.Obstacle
    - pylot.perception.tracking.multi_object_tracker.MultiObjectTracker
    - pylot.perception.tracking.deep_sort_tracker.MultiObjectDeepSORTTracker
    - pylot.perception.tracking.sort_tracker.MultiObjectSORTTracker

- External dependencies
    - cvxpy.expressions.constants
    - erdos.Message
    - erdos.Timestamp
    - torch
    - nanonets_object_tracking.deepsort.deepsort_rbc
    - sort.sort.Sort
    - carla.Location
    - carla.Vector3D
    - carla.Transform
    - carla.BoundingBox
    - shapely.geometry.LineString
    - cv2
    - pygame
    - open3d
'''

# send pose + waypoints data, receive messages.ControlMessage
class ControllerInput():
    def __init__(self, pose_msg, waypoints_msg, type):
        self.pose = pose_msg
        self.waypoints = waypoints_msg
        self.type = type

# send frame + obstacle message, receive messages.ObstacleMessage
class TrackerInput():
    def __init__(self, frame_msg, obstacle_msg, runtime, type):
        self.frame_msg = frame_msg
        self.obstacles = obstacle_msg.obstacles
        self.runtime = runtime
        self.type = type

# TODO: 
# - write client code in operators to call socket service
# - add flags and params for server / client endpoints
# - test in isolation (Mac, Pi, AWS) and then integration (E2E)

# Set up socket server for remote control and tracker
# Control - 
#   input=(type, Pose, WaypointMessage)
#   output=(ControlMessage, error)
# Tracker -
#   input=(type, CameraFrame, ObstacleMessage)
#   output=(ObstacleMessage, error)
