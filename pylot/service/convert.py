import pylot.service.objects as objects
import pylot.service.utils as utils
import pylot.service.messages as messages

import pylot.utils
import pylot.perception
import pylot.planning.waypoints
import pylot.control.messages

import pylot.perception.detection.obstacle
import pylot.perception.detection.utils
import pylot.perception.messages

from collections import deque

def to_pylot_transform(transform: utils.Transform):
    return pylot.utils.Transform(
        location=to_pylot_location(transform.location),
        rotation=to_pylot_rotation(transform.rotation),
        matrix=transform.matrix
    )

def from_pylot_transform(transform: pylot.utils.Transform):
    return utils.Transform(
        location=from_pylot_location(transform.location),
        rotation=from_pylot_rotation(transform.rotation),
        matrix=transform.matrix
    )

def to_pylot_location(location: utils.Location):
    return pylot.utils.Location(location.x, location.y, location.z)

def from_pylot_location(location: pylot.utils.Location):
    return utils.Location(location.x, location.y, location.z)

def to_pylot_rotation(rotation: utils.Rotation):
    return pylot.utils.Rotation(rotation.pitch, rotation.yaw, rotation.roll)

def from_pylot_rotation(rotation: pylot.utils.Rotation):
    return utils.Rotation(rotation.pitch, rotation.yaw, rotation.roll)

def to_pylot_vector(vector: utils.Vector3D):
    return pylot.utils.Vector3D(vector.x, vector.y, vector.z)

def from_pylot_vector(vector: pylot.utils.Vector3D):
    return utils.Vector3D(vector.x, vector.y, vector.z)

def to_pylot_boundingbox(bb: objects.BoundingBox2D):
    return pylot.perception.detection.utils.BoundingBox2D(bb.x_min, bb.x_max, bb.y_min, bb.y_max)

def from_pylot_boundingbox(bb: pylot.perception.detection.utils.BoundingBox2D):
    return objects.BoundingBox2D(bb.x_min, bb.x_max, bb.y_min, bb.y_max)

def to_pylot_obstacle(ob: objects.Obstacle):
    return pylot.perception.detection.obstacle.Obstacle(
        bounding_box=to_pylot_boundingbox(ob.bounding_box),
        confidence=ob.confidence,
        label=ob.label,
        id=ob.id,
        transform=to_pylot_transform(ob.transform),
        detailed_label=ob.detailed_label,
        bounding_box_2D=to_pylot_boundingbox(ob.bounding_box_2D)
    )

def from_pylot_obstacle(ob: pylot.perception.detection.obstacle.Obstacle):
    return objects.Obstacle(
        bounding_box=from_pylot_boundingbox(ob.bounding_box),
        confidence=ob.confidence,
        label=ob.label,
        id=ob.id,
        transform=from_pylot_transform(ob.transform),
        detailed_label=ob.detailed_label,
        bounding_box_2D=from_pylot_boundingbox(ob.bounding_box_2D)
    )

def to_pylot_pose(pose: utils.Pose):
    return pylot.utils.Pose(
        transform=to_pylot_transform(pose.transform),
        forward_speed=pose.forward_speed,
        velocity_vector=to_pylot_vector(pose.velocity_vector),
        localization_time=pose.localization_time
    )

def from_pylot_pose(pose: pylot.utils.Pose):
    return utils.Pose(
        transform=from_pylot_transform(pose.transform),
        forward_speed=pose.forward_speed,
        velocity_vector=from_pylot_vector(pose.velocity_vector),
        localization_time=pose.localization_time
    )

def to_pylot_waypoint(wps: objects.Waypoints):
    return pylot.planning.waypoints.Waypoints(
        waypoints=deque([to_pylot_transform(t) for t in wps.waypoints]),
        target_speeds=wps.target_speeds
    )

def from_pylot_waypoint(wps: pylot.planning.waypoints.Waypoints):
    return objects.Waypoints(
        waypoints=deque([from_pylot_transform(t) for t in wps.waypoints]),
        target_speeds=wps.target_speeds
    )

def to_pylot_pose_message():
    return erdos.Message()

def to_pylot_control_message(cm: messages.ControlMessage, ts):
    return pylot.control.messages.ControlMessage(
        steer=cm.steer,
        throttle=cm.throttle,
        brake=cm.brake,
        hand_brake=cm.hand_brake,
        reverse=cm.reverse,
        timestamp=ts
    )

def to_pylot_obstacle_message(om: messages.ObstaclesMessage, ts, rt):
    return pylot.perception.messages.ObstaclesMessage(
        timestamp=ts,
        obstacles=[to_pylot_obstacle(o) for o in om.obstacles],
        runtime=rt
    )
