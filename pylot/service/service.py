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

import struct

def send_msg(sock, msg):
    # Prefix each message with a 4-byte length (network byte order)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def recv_msg(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    return recvall(sock, msglen)

def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

# send pose + waypoints data, receive messages.ControlMessage
class ControllerInput():
    def __init__(self, pose_msg, waypoints_msg, type):
        self.pose = pose_msg
        self.waypoints = waypoints_msg
        self.type = type

# send frame + obstacle message, receive messages.ObstacleMessage
class TrackerInput():
    def __init__(self, frame, obstacles, reinit, type):
        self.frame = frame
        self.obstacles = obstacles
        self.reinit = True
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
