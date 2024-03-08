from collections import defaultdict, deque

import params
import copy
import numpy as np

from objects.objects import Transform, Location, Rotation, RGBCameraSetup, ObstacleTrajectory, Vector2D
from objects.frames import DepthFrame, PointCloud

_camera_transform = Transform(Location(1.3, 0.0, 1.8), Rotation(pitch=-15))
_camera_setup = RGBCameraSetup('center_camera', params.camera_image_width, params.camera_image_height, _camera_transform, params.camera_fov)

class ObstacleLocationHistory:

    def __init__(self):
        self._obstacle_history = defaultdict(deque)

    def get_location_history(self, timestamp, pose, depth_frame, obstacles):
        vehicle_transform = pose.transform
        obstacles_with_location = self._get_obstacle_locations(obstacles, depth_frame, vehicle_transform)

        ids_cur_timestamp = []
        obstacle_trajectories = []
        for obstacle in obstacles_with_location:
            # Ignore obstacles that are far away.
            if (vehicle_transform.location.distance(obstacle.transform.location) > params.dynamic_obstacle_distance_threshold):
                continue
            ids_cur_timestamp.append(obstacle.id)
            self._obstacle_history[obstacle.id].append(obstacle)
            # Transform obstacle location from global world coordinates to ego-centric coordinates.
            cur_obstacle_trajectory = []
            for obstacle in self._obstacle_history[obstacle.id]:
                new_location = \
                    vehicle_transform.inverse_transform_locations(
                        [obstacle.transform.location])[0]
                cur_obstacle_trajectory.append(Transform(new_location, Rotation()))
            # The trajectory is relative to the current location.
            obstacle_trajectories.append(ObstacleTrajectory(obstacle, cur_obstacle_trajectory))
            
        return timestamp, obstacle_trajectories


    def _get_obstacle_locations(self, obstacles, depth_frame, ego_transform):
        if isinstance(depth_frame, PointCloud):
            point_cloud = depth_frame
            # Get the position of the camera in world frame of reference.
            transformed_camera_setup = copy.deepcopy(_camera_setup)
            transformed_camera_setup.set_transform(
                ego_transform * transformed_camera_setup.transform)

            obstacles_with_location = []
            for obstacle in obstacles:
                location = point_cloud.get_pixel_location(
                    obstacle.bounding_box_2D.get_center_point(),
                    transformed_camera_setup)
                if location is not None:
                    obstacle.transform = Transform(
                        location, Rotation())
                    obstacles_with_location.append(obstacle)
            return obstacles_with_location
        elif isinstance(depth_frame, DepthFrame):
            depth_frame.camera_setup.set_transform(
                ego_transform * depth_frame.camera_setup.transform)

            for obstacle in obstacles:
                center_point = obstacle.bounding_box_2D.get_center_point()
                # Sample several points around the center of the bounding box
                # in case the bounding box is not well centered on the obstacle.
                # In such situations the center point might be in between legs,
                # and thus we might overestimate the distance.
                sample_points = []
                for delta_x in range(-30, 30, 5):
                    for delta_y in range(-30, 30, 5):
                        sample_point = center_point + Vector2D(delta_x, delta_y)
                        if obstacle.bounding_box.is_within(sample_point):
                            sample_points.append(sample_point)
                locations = depth_frame.get_pixel_locations(sample_points)
                # Choose the closest from the locations of the sampled points.
                min_distance = np.infty
                closest_location = None
                for location in locations:
                    dist = location.distance(ego_transform.location)
                    if dist < min_distance:
                        min_distance = dist
                        closest_location = location
                obstacle.transform = Transform(closest_location, Rotation())
            return obstacles
        else:
            raise ValueError('Unexpected depth message type')
