"""This module implements perception related Message classes."""

import numpy as np
import time

import objects.frames
from objects.objects import Transform, Location, Rotation


class FrameMessage():
    """Message class to be used to send camera frames.

    Args:
        timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of the
            message.
        frame (:py:class:`~pylot.perception.camera_frame.CameraFrame`): The
            camera frame.

    Attributes:
        frame (:py:class:`~pylot.perception.camera_frame.CameraFrame`): The
            camera frame.
    """
    def __init__(self, timestamp, frame):
        super(FrameMessage, self).__init__(timestamp, None)
        if not isinstance(frame, objects.frames.CameraFrame):
            raise ValueError(
                'frame should be of type perception.camera_frame.CameraFrame')
        self.frame = frame

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'FrameMessage(timestamp: {}, frame: {})'.format(
            self.timestamp, self.frame)


class DepthFrameMessage():
    """Message class to be used to send depth camera frames.

    Args:
        timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of the
            message.
        frame (:py:class:`~pylot.perception.depth_frame.DepthFrame`): The
            depth frame.

    Attributes:
        frame (:py:class:`~pylot.perception.depth_frame.DepthFrame`): The
            depth frame.
    """
    def __init__(self, timestamp, frame):
        super(DepthFrameMessage, self).__init__(timestamp, None)
        if not isinstance(frame, objects.frames.DepthFrame):
            raise ValueError(
                'frame should be of type perception.depth_frame.DepthFrame')
        self.frame = frame

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'DepthMessage(timestamp: {}, depth_frame: {})'.format(
            self.timestamp, self.frame)


class SegmentedFrameMessage():
    """This class represents a message to be used to send segmented frames

    Args:
        timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of the
            message.
        frame (:py:class:`~.segmentation.segmented_frame.SegmentedFrame`):
            The segmented frame.
        runtime (:obj:`float`, optional): The runtime of the operator that
            produced the segmented frame (in ms).

    Attributes:
        frame (:py:class:`~.segmentation.segmented_frame.SegmentedFrame`):
            The segmented frame.
        runtime (:obj:`float`): The runtime of the operator that produced the
            segmented frame (in ms).
    """
    def __init__(self, timestamp, frame, runtime=0):
        super(SegmentedFrameMessage, self).__init__(timestamp, None)
        if not isinstance(frame, objects.frames.SegmentedFrame):
            raise ValueError('frame should be of type SegmentedFrame')
        self.frame = frame
        self.runtime = runtime

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'SegmentedFrameMessage(timestamp {}, frame: {})'.format(
            self.timestamp, self.frame)


class PointCloudMessage():
    """Message class to be used to send point clouds.

    Args:
        timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of the
            message.
        pc (:py:class:`~pylot.perception.point_cloud.PointCloud`): The point
            cloud to send.

    Attributes:
        point_cloud (:py:class:`~pylot.perception.point_cloud.PointCloud`): The
            point cloud.
    """
    def __init__(self, timestamp, pc):
        super(PointCloudMessage, self).__init__(timestamp, None)
        if not isinstance(pc, objects.frames.PointCloud):
            raise ValueError(
                'pc should be of type perception.point_cloud.PointCloud')
        self.point_cloud = pc

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'PointCloudMessage(timestamp: {}, point cloud: {})'.format(
            self.timestamp, self.point_cloud)


class ObstaclesMessage():
    """Used to send detected obstacles.

    Args:
        timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of the
            message.
        obstacles (list(:py:class:`~.detection.obstacle.Obstacle`)):
            Detected obstacles.
        runtime (:obj:`float`, optional): The runtime of the operator that
            produced the obstacles (in ms).


    Attributes:
        obstacles (list(:py:class:`~.detection.obstacle.Obstacle`)):
            Detected obstacles.
        runtime (:obj:`float`, optional): The runtime of the operator that
            produced the obstacles (in ms).
    """
    def __init__(self, timestamp, obstacles, runtime=0):
        super(ObstaclesMessage, self).__init__(timestamp, None)
        self.obstacles = obstacles
        self.runtime = runtime

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'ObstaclesMessage(timestamp: {}, runtime: {}, '\
            'obstacles: {})'.format(
                self.timestamp, self.runtime, self.obstacles)


class ObstaclePositionsSpeedsMessage():
    """Represents a message to be used to send vehicle position and speed info.

    Args:
        obstacle_positions_speeds: A list of 2D obstacle positions.
        timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of the
            message.

    Attributes:
        obstacle_positions_speeds: A list of 2D obstacle positions.
    """
    def __init__(self, timestamp, obstacle_positions_speeds):
        super(ObstaclePositionsSpeedsMessage, self).__init__(timestamp, None)
        self.obstacle_positions_speeds = obstacle_positions_speeds

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return ('ObstaclePositionsSpeedsMessage(timestamp {}, '
                'positions: {})'.format(self.timestamp,
                                        self.obstacle_positions_speeds))


class ObstacleTrajectoriesMessage():
    """Message to be used to send obstacle trajectory info."""

    def __init__(self, obstacle_trajectories):
        self.obstacle_trajectories = obstacle_trajectories

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        trajectories_str = '\n'.join(
            [str(traj) for traj in self.obstacle_trajectories])
        return ('ObstacleTrajectoriesMessage('
                'trajectories: {})'.format(trajectories_str))

    def get_nearby_obstacles_info(self, radius, filter_fn=None):
        """Gets a lost of obstacle that are within the radius.

        Using the list of obstacle trajectories in the message (which are
        in the ego-vehicle's frame of reference), return a list of obstacles
        that are within a specified radius of the ego-vehicle, as well as
        a list of their transforms, sorted by increasing distance.

        Args:
            radius: Filter obstacle trajectories outside the radius.
            filter_fn: Function to filter obstacle trajectories.
        """
        if filter_fn:
            filtered_trajectories = list(
                filter(filter_fn, self.obstacle_trajectories))
        else:
            filtered_trajectories = self.obstacle_trajectories
        distances = [
            v.trajectory[-1].get_angle_and_magnitude(Location())[1]
            for v in filtered_trajectories
        ]
        sorted_trajectories = [
            v for v, d in sorted(zip(filtered_trajectories, distances),
                                 key=lambda pair: pair[1]) if d <= radius
        ]

        if len(sorted_trajectories) == 0:
            return sorted_trajectories, []

        nearby_obstacles_ego_locations = np.stack(
            [t.trajectory[-1] for t in sorted_trajectories])
        nearby_obstacles_ego_transforms = []

        # Add appropriate rotations to nearby_obstacles_ego_transforms, which
        # we estimate using the direction determined by the last two distinct
        # locations
        for i in range(len(sorted_trajectories)):
            cur_obstacle_angle = sorted_trajectories[
                i].estimate_obstacle_orientation()
            nearby_obstacles_ego_transforms.append(
                Transform(location=nearby_obstacles_ego_locations[i].location,
                          rotation=Rotation(yaw=cur_obstacle_angle)))
        return sorted_trajectories, nearby_obstacles_ego_transforms


class LanesMessage():
    """Message to be used to send info about lanes.

    Args:
        timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of the
            message.
        lanes (list(:py:class:`~.detection.lane.Lane`)): Lis of lanes.

    Attributes:
        lanes (list(:py:class:`~.detection.lane.Lane`)): List of lanes.
    """
    def __init__(self, timestamp, lanes):
        super(LanesMessage, self).__init__(timestamp, None)
        self.lanes = lanes

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'LaneMessage(timestamp: {}, lanes: {})'.format(
            self.timestamp, self.lanes)


class TrafficLightsMessage():
    """Message class to be used to send info about traffic lights.

    Args:
        timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of the
            message.
        traffic_lights (list(:py:class:`~.detection.traffic_light.TrafficLight`)):  # noqa: E501
            A list of traffic lights.

    Attributes:
        traffic_lights (list(:py:class:`~.detection.traffic_light.TrafficLight`)):
            A list of traffic lights.
    """
    def __init__(self, timestamp, traffic_lights):
        super(TrafficLightsMessage, self).__init__(timestamp, None)
        self.obstacles = traffic_lights

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'TrafficLightsMessage(timestamp: {}, ' \
            'traffic lights: {})'.format(
                self.timestamp, self.obstacles)


class StopSignsMessage():
    """Message class to be used to send info about stop signs.

    Args:
        timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of the
            message.
        stop_signs (list(:py:class:`~.detection.stop_sign.StopSign`)):
            A list of stop signs.

    Attributes:
        stop_signs (list(:py:class:`~.detection.stop_sign.StopSign`)):
            A list of stop signs.
    """
    def __init__(self, timestamp, stop_signs):
        super(StopSignsMessage, self).__init__(timestamp, None)
        self.stop_signs = stop_signs

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'StopSignsMessage(timestamp: {}, stop signs: {})'.format(
            self.timestamp, self.stop_signs)


class SpeedSignsMessage():
    """Message class to be used to send info about speed limit actors.

    Args:
        timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of the
            message.
        speed_signs (list(:py:class:`~.detection.speed_limit_sign.SpeedLimitSign`)):  # noqa: E501
            A list of speed limit signs.

    Attributes:
        speed_signs (list(:py:class:`~.detection.speed_limit_sign.SpeedLimitSign`)):
            A list of speed limit signs.
    """
    def __init__(self, timestamp, speed_signs):
        super(SpeedSignsMessage, self).__init__(timestamp, None)
        self.speed_signs = speed_signs

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'SpeedSignsMessage(timestamp: {}, speed signs: {})'.format(
            self.timestamp, self.speed_signs)


class ControlMessage():
    """This class represents a message to be used to send control commands.

    Args:
        steer (:obj:`float`): Steer angle between [-1.0, 1.0].
        throttle (:obj:`float`): Throttle command between [0.0, 1.0].
        brake (:obj:`float`): Brake command between [0.0, 1.0].
        hand_brake (bool): Boolean controlling hand-brake engagement.
        reverse (bool): Boolean controlling reverse gear engagement.
        timestamp (:py:class:`erdos.timestamp.Timestamp`): The timestamp of
            the message.
    """
    def __init__(self, steer: float, throttle: float, brake: float,
                 hand_brake: bool, reverse: bool, timestamp, local_send_time = None):
        # assert steer >= -1 and steer <= 1, 'Steer angle must be in [-1, 1]'
        self.steer = steer
        # assert throttle >= 0 and throttle <= 1, 'Throttle must be in [0, 1]'
        self.throttle = throttle
        # assert brake >= 0 and brake <= 1, 'Brake must be in [0, 1]'
        self.brake = brake
        self.hand_brake = hand_brake
        self.reverse = reverse
        self.creation_time = time.time()
        self.local_send_time = local_send_time

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return ('ControlMessage(timestamp: {}, steer: {}, throttle: {}, '
                'brake: {}, hand_brake: {}, reverse: {})'.format(
                    self.timestamp, self.steer, self.throttle, self.brake,
                    self.hand_brake, self.reverse))


class SensorMessage():
    def __init__(self, timestamp, frame, depth_frame, pose, local_send_time=None):
        self.timestamp = timestamp
        self.frame = frame
        self.depth_frame = depth_frame
        self.pose = pose
        self.local_send_time = local_send_time

class PlannerMessage():
    def __init__(self, pose, waypoints):
        self.pose = pose
        self.waypoints = waypoints