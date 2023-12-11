import numpy as np

import objects

class ObstaclesMessage(object):
    """Used to send detected obstacles.

    Args:
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
    def __init__(self, obstacles, runtime=0):
        #super(ObstaclesMessage, self).__init__(None)
        self.obstacles = obstacles
        self.runtime = runtime

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'ObstaclesMessage(runtime: {}, '\
            'obstacles: {})'.format(
                self.runtime, self.obstacles)


class DepthFrameMessage(object):
    """Message class to be used to send depth camera frames.

    Args:
        frame (:py:class:`~frames.DepthFrame`): The
            depth frame.

    Attributes:
        frame (:py:class:`~frames.DepthFrame`): The
            depth frame.
    """
    def __init__(self, frame):
        #super(DepthFrameMessage, self).__init__(None)
        if not isinstance(frame, objects.DepthFrame):
            raise ValueError('frame should be of type perception.depth_frame.DepthFrame')
        self.frame = frame

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'DepthMessage(depth_frame: {})'.format(self.frame)


class PointCloudMessage(object):
    """Message class to be used to send point clouds.

    Args:
        pc (:py:class:`~pylot.perception.point_cloud.PointCloud`): The point
            cloud to send.

    Attributes:
        point_cloud (:py:class:`~pylot.perception.point_cloud.PointCloud`): The
            point cloud.
    """
    def __init__(self, pc):
        #super(PointCloudMessage, self).__init__(None)
        if not isinstance(pc, objects.PointCloud):
            raise ValueError(
                'pc should be of type perception.point_cloud.PointCloud')
        self.point_cloud = pc

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'PointCloudMessage(point cloud: {})'.format(self.point_cloud)


class ControlMessage():
    """This class represents a message to be used to send control commands.

    Args:
        steer (:obj:`float`): Steer angle between [-1.0, 1.0].
        throttle (:obj:`float`): Throttle command between [0.0, 1.0].
        brake (:obj:`float`): Brake command between [0.0, 1.0].
        hand_brake (bool): Boolean controlling hand-brake engagement.
        reverse (bool): Boolean controlling reverse gear engagement.
    """
    def __init__(self, steer: float, throttle: float, brake: float,
                 hand_brake: bool, reverse: bool):
        #super(ControlMessage, self).__init__(None)
        assert steer >= -1 and steer <= 1, 'Steer angle must be in [-1, 1]'
        self.steer = steer
        assert throttle >= 0 and throttle <= 1, 'Throttle must be in [0, 1]'
        self.throttle = throttle
        assert brake >= 0 and brake <= 1, 'Brake must be in [0, 1]'
        self.brake = brake
        self.hand_brake = hand_brake
        self.reverse = reverse
        #self.creation_time = time.time()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return ('ControlMessage(steer: {}, throttle: {}, '
                'brake: {}, hand_brake: {}, reverse: {})'.format(
                    self.steer, self.throttle, self.brake,
                    self.hand_brake, self.reverse))


class WaypointsMessage():
    """Message class to be used to send waypoints.

    Optionally can also send a target speed for each waypoint.

    """
    def __init__(self,
                 waypoints,
                 agent_state=None):
        #super(WaypointsMessage, self).__init__(None)
        self.waypoints = waypoints
        self.agent_state = agent_state

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'WaypointMessage(waypoints: {}, '\
            'agent_state: {}'.format(self.waypoints, self.agent_state)