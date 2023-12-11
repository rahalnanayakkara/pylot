import numpy as np

import csv
import itertools
from collections import deque

from utils import Location, Transform, Rotation, RoadOption
from detection_utils import BoundingBox2D, BoundingBox3D, get_bounding_box_in_camera_view

VEHICLE_LABELS = {'car', 'bicycle', 'motorcycle', 'bus', 'truck', 'vehicle'}

class Obstacle(object):
    """Class used to store info about obstacles.

    This class provides helper functions to detect obstacles and provide
    bounding boxes for them.

    Args:
        bounding_box (:py:class:`.BoundingBox2D`): The bounding box of the
            obstacle (can be 2D or 3D).
        confidence (:obj:`float`): The confidence of the detection.
        label (:obj:`str`): The label of the obstacle.
        id (:obj:`int`): The identifier of the obstacle.
        transform (:py:class:`~Transform`, optional): Transform of
            the obstacle in the world.

    Attributes:
        bounding_box (:py:class:`~pylot.utisl.BoundingBox2D`): Bounding box of
            the obstacle (can be 2D or 3D).
        confidence (:obj:`float`): The confidence of the detection.
        label (:obj:`str`): The label of the obstacle.
        id (:obj:`int`): The identifier of the obstacle.
        transform (:py:class:`~Transform`): Transform of the
            obstacle.
    """
    def __init__(self,
                 bounding_box,
                 confidence: float,
                 label: str,
                 id: int = -1,
                 transform: Transform = None,
                 detailed_label: str = '',
                 bounding_box_2D: BoundingBox2D = None):
        self.bounding_box = bounding_box
        if isinstance(bounding_box, BoundingBox2D):
            self.bounding_box_2D = bounding_box
        else:
            self.bounding_box_2D = bounding_box_2D
        self.confidence = confidence
        self.label = label
        self.id = id
        self.transform = transform
        self.detailed_label = detailed_label
        if label == 'vehicle':
            self.segmentation_class = 10
        elif label == 'person':
            self.segmentation_class = 4
        else:
            self.segmentation_class = None
        # Thresholds to be used for detection of the obstacle.
        self.__segmentation_threshold = 0.20
        self.__depth_threshold = 5

    @classmethod
    def from_simulator_actor(cls, actor):
        """Creates an Obstacle from a simulator actor.

        Args:
            actor: The actor to initialize the obstacle with.

        Returns:
            :py:class:`.Obstacle`: An obstacle instance.
        """
        # from carla import Vehicle, Walker
        # if not isinstance(actor, (Vehicle, Walker)):
        #     raise ValueError("The actor should be of type Vehicle or "
        #                      "Walker to initialize the Obstacle class.")
        # We do not use everywhere from_simulator* methods in order to reduce
        # runtime.
        # Convert the transform provided by the simulation to the Pylot class.
        transform = Transform.from_simulator_transform(
            actor.get_transform())
        # Convert the bounding box from the simulation to the Pylot one.
        bounding_box = BoundingBox3D.from_simulator_bounding_box(
            actor.bounding_box)
        if isinstance(actor, Vehicle):
            label = 'vehicle'
        else:
            label = 'person'
        # Get the simulator actor from type_id (e.g. vehicle.ford.mustang).
        detailed_label = actor.type_id
        # TODO (Sukrit): Move from vehicles and people to separate classes
        # for bicycles, motorcycles, cars and persons.
        return cls(bounding_box, 1.0, label, actor.id, transform,
                   detailed_label)

    def as_mot16_str(self, timestamp):
        if not self.bounding_box_2D:
            raise ValueError(
                'Obstacle {} does not have 2D bounding box'.format(self.id))
        log_line = "{},{},{},{},{},{},{},{},{},{}\n".format(
            timestamp, self.id, self.bounding_box_2D.x_min,
            self.bounding_box_2D.y_min, self.bounding_box_2D.get_width(),
            self.bounding_box_2D.get_height(), 1.0, -1, -1, -1)
        return log_line

    def _distance(self, other_transform: Transform):
        """Computes the distance from the obstacle to the other transform.

        The distance provides an estimate of the depth returned by the depth
        camera sensor in the simulator. As a result, the distance is defined
        as the displacement of the obstacle along either the X or the Y axis.

        Args:
            other_transform (:py:class:`~Transform`): The other
                transform.

        Returns:
            :obj:`float`: The distance (in metres) of the obstacle from the
            transform.
        """
        import numpy as np
        if self.transform is None:
            raise ValueError('Obstacle {} does not have a transform'.format(
                self.id))
        # Get the location of the vehicle and the obstacle as numpy arrays.
        other_location = other_transform.location.as_numpy_array()
        obstacle_location = self.transform.location.as_numpy_array()

        # Calculate the vector from the vehicle to the obstacle.
        # Scale it by the forward vector, and calculate the norm.
        relative_vector = other_location - obstacle_location
        distance = np.linalg.norm(
            relative_vector * other_transform.forward_vector.as_numpy_array())
        return distance

    def draw_on_frame(self,
                      frame,
                      bbox_color_map,
                      ego_transform: Transform = None,
                      text: str = None):
        """Annotate the image with the bounding box of the obstacle."""
        if text is None:
            text = '{}, {:.1f}'.format(self.label, self.confidence)
            if self.id != -1:
                text += ', id:{}'.format(self.id)
            if ego_transform is not None and self.transform is not None:
                text += ', {:.1f}m'.format(
                    ego_transform.location.distance(self.transform.location))
        if self.label in bbox_color_map:
            color = bbox_color_map[self.label]
        else:
            color = [255, 255, 255]
        # Show bounding box.
        if self.bounding_box_2D:
            # Draw the 2D bounding box if available.
            frame.draw_box(self.bounding_box_2D.get_min_point(),
                           self.bounding_box_2D.get_max_point(), color)
            frame.draw_text(self.bounding_box_2D.get_min_point(), text, color)
        elif isinstance(self.bounding_box, BoundingBox3D):
            if self.bounding_box.corners is None:
                raise ValueError(
                    'Obstacle {} does not have bbox corners'.format(self.id))
            corners = self.bounding_box.to_camera_view(
                None, frame.camera_setup.get_extrinsic_matrix(),
                frame.camera_setup.get_intrinsic_matrix())
            frame.draw_3d_box(corners, color)
        else:
            raise ValueError('Obstacle {} does not have bounding box'.format(
                self.id))

    def draw_trajectory_on_frame(self,
                                 trajectory,
                                 frame,
                                 point_color,
                                 draw_label: bool = False):
        # Intrinsic and extrinsic matrix of the top down camera.
        extrinsic_matrix = frame.camera_setup.get_extrinsic_matrix()
        intrinsic_matrix = frame.camera_setup.get_intrinsic_matrix()
        if isinstance(self.bounding_box, BoundingBox3D):
            # Draw bounding boxes.
            start_location = self.bounding_box.transform.location - \
                self.bounding_box.extent
            end_location = self.bounding_box.transform.location + \
                self.bounding_box.extent
            for transform in trajectory:
                [start_transform,
                 end_transform] = transform.transform_locations(
                     [start_location, end_location])
                start_point = start_transform.to_camera_view(
                    extrinsic_matrix, intrinsic_matrix)
                end_point = end_transform.to_camera_view(
                    extrinsic_matrix, intrinsic_matrix)
                if frame.in_frame(start_point) or frame.in_frame(end_point):
                    frame.draw_box(start_point, end_point, point_color)
        else:
            # Draw points.
            for transform in trajectory:
                screen_point = transform.location.to_camera_view(
                    extrinsic_matrix, intrinsic_matrix)
                if frame.in_frame(screen_point):
                    # Draw trajectory on frame.
                    frame.draw_point(screen_point, point_color)
        if draw_label and len(trajectory) > 0:
            text = '{}, {}'.format(self.label, self.id)
            screen_point = trajectory[-1].location.to_camera_view(
                extrinsic_matrix, intrinsic_matrix)
            frame.draw_text(screen_point, text, point_color)

    def get_bounding_box_corners(self,
                                 obstacle_transform,
                                 obstacle_radius=None):
        """Gets the corners of the obstacle's bounding box.
        Note:
            The bounding box is applied on the given obstacle transfom, and not
            on the default obstacle transform.
        """
        # Use 3d bounding boxes if available, otherwise use default
        if isinstance(self.bounding_box, BoundingBox3D):
            start_location = (self.bounding_box.transform.location -
                              self.bounding_box.extent)
            end_location = (self.bounding_box.transform.location +
                            self.bounding_box.extent)
            [start_location,
             end_location] = obstacle_transform.transform_locations(
                 [start_location, end_location])
        else:
            obstacle_radius_loc = Location(obstacle_radius,
                                                       obstacle_radius)
            start_location = obstacle_transform.location - obstacle_radius_loc
            end_location = obstacle_transform.location + obstacle_radius_loc
        return [
            min(start_location.x, end_location.x),
            min(start_location.y, end_location.y),
            max(start_location.x, end_location.x),
            max(start_location.y, end_location.y)
        ]

    def get_in_log_format(self):
        if not self.bounding_box_2D:
            raise ValueError(
                'Obstacle {} does not have 2D bounding box'.format(self.id))
        min_point = self.bounding_box_2D.get_min_point()
        max_point = self.bounding_box_2D.get_max_point()
        return (self.label, self.detailed_label, self.id,
                ((min_point.x, min_point.y), (max_point.x, max_point.y)))

    def is_animal(self):
        return self.label in [
            'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe'
        ]

    def is_person(self):
        return self.label == 'person'

    def is_speed_limit(self):
        return self.label in [
            'speed limit 30', 'speed limit 60', 'speed limit 90'
        ]

    def is_stop_sign(self):
        return self.label == 'stop sign' or self.label == 'stop marking'

    def is_traffic_light(self):
        return self.label in [
            'red traffic light', 'yellow traffic light', 'green traffic light',
            'off traffic light'
        ]

    def is_vehicle(self):
        # Might want to include train.
        return self.label in VEHICLE_LABELS

    def populate_bounding_box_2D(self, depth_frame, segmented_frame):
        """Populates the 2D bounding box for the obstacle.

        Heuristically uses the depth frame and segmentation frame to figure out
        if the obstacle is in view of the camera or not.

        Args:
            depth_frame (:py:class:`~pylot.perception.depth_frame.DepthFrame`):
                Depth frame used to compare the depth to the distance of the
                obstacle from the sensor.
            segmented_frame (:py:class:`~pylot.perception.segmentation.segmented_frame.SegmentedFrame`):  # noqa: E501
                Segmented frame used to refine the conversions.

        Returns:
            :py:class:`~pylot.utisl.BoundingBox2D`: An instance representing a
            rectangle over the obstacle if the obstacle is deemed to be
            visible, None otherwise.
        """
        if self.bounding_box_2D:
            return self.bounding_box_2D
        # Convert the bounding box of the obstacle to the camera coordinates.
        bb_coordinates = self.bounding_box.to_camera_view(
            self.transform, depth_frame.camera_setup.get_extrinsic_matrix(),
            depth_frame.camera_setup.get_intrinsic_matrix())

        # Threshold the bounding box to be within the camera view.
        bbox_2d = get_bounding_box_in_camera_view(
            bb_coordinates, depth_frame.camera_setup.width,
            depth_frame.camera_setup.height)
        if not bbox_2d:
            return None
        # Crop the segmented and depth image to the given bounding box.
        cropped_image = segmented_frame.as_numpy_array()[
            bbox_2d.y_min:bbox_2d.y_max, bbox_2d.x_min:bbox_2d.x_max]
        cropped_depth = depth_frame.as_numpy_array()[
            bbox_2d.y_min:bbox_2d.y_max, bbox_2d.x_min:bbox_2d.x_max]

        # If the size of the bounding box is greater than 0, ensure that the
        # bounding box contains more than a threshold of pixels corresponding
        # to the required segmentation class.
        if cropped_image.size > 0:
            masked_image = np.zeros_like(cropped_image)
            masked_image[np.where(
                cropped_image == self.segmentation_class)] = 1
            seg_threshold = self.__segmentation_threshold * masked_image.size
            if np.sum(masked_image) >= seg_threshold:
                # The bounding box contains the required number of pixels that
                # belong to the required class. Ensure that the depth of the
                # obstacle is the depth in the image.
                masked_depth = cropped_depth[np.where(masked_image == 1)]
                mean_depth = np.mean(masked_depth) * 1000
                depth = self._distance(
                    depth_frame.camera_setup.get_transform())
                if abs(depth - mean_depth) <= self.__depth_threshold:
                    self.bounding_box_2D = bbox_2d
                    return bbox_2d
        return None

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        obstacle = 'Obstacle(id: {}, label: {}, confidence: {}, '\
            'bbox: {})'.format(self.id, self.label, self.confidence,
                               self.bounding_box)
        if self.transform:
            return obstacle + ' at ' + str(self.transform)
        else:
            return obstacle

class Waypoints(object):
    """Stores waypoints and provides methods to wrangle them."""
    def __init__(self, waypoints, target_speeds=None, road_options=None):
        self.waypoints = waypoints
        if target_speeds is None:
            # Set target speed to 0 if it is not specified.
            self.target_speeds = deque([0 for _ in range(len(waypoints))])
        else:
            self.target_speeds = target_speeds
        self.road_options = road_options

    @classmethod
    def read_from_csv_file(cls, csv_file_name: str, target_speed):
        """Reads waypoints from a csv file."""
        csv_file = open(csv_file_name)
        csv_reader = csv.reader(csv_file)
        waypoints = []
        for row in csv_reader:
            x = float(row[0])
            y = float(row[1])
            z = float(row[2])
            waypoint = Transform(Location(x, y, z),Rotation(0, 0, 0))
            waypoints.append(waypoint)
        target_speeds = deque([target_speed for _ in range(len(waypoints))])
        return cls(deque(waypoints), target_speeds)

    def apply_speed_factor(self, speed_factor: float):
        if self.target_speeds:
            self.target_speeds = [
                speed_factor * ts for ts in self.target_speeds
            ]

    def as_numpy_array_2D(self):
        """Returns the waypoints as a numpy array of lists of x and y."""
        wx = []
        wy = []
        for wp in self.waypoints:
            wx.append(wp.location.x)
            wy.append(wp.location.y)
        return np.array([wx, wy])

    def closest_waypoint(self, location: Location):
        """Finds the closest waypoint to the location."""
        min_dist = np.infty
        min_index = 0
        for index, waypoint in enumerate(self.waypoints):
            dist = waypoint.location.distance(location)
            if dist < min_dist:
                min_dist = dist
                min_index = index
        return min_index

    def is_empty(self) -> bool:
        return len(self.waypoints) == 0

    def remove_waypoint_if_close(self,
                                 location: Location,
                                 distance: float = 5) -> bool:
        """Removes the first waypoint if it is less than distance m away."""
        if self.waypoints is None or len(self.waypoints) == 0:
            return False
        if location.distance(self.waypoints[0].location) < distance:
            self.waypoints.popleft()
            if self.target_speeds:
                self.target_speeds.popleft()
            if self.road_options:
                self.road_options.popleft()
            return True
        return False

    def remove_completed(self,
                         location: Location,
                         ego_transform: Transform = None):
        """Removes waypoints that the ego vehicle has already completed.

        The method first finds the closest waypoint to the location,
        and then removes all waypoints that are before the closest waypoint.

        Args:
            location (:py:class:`Location`): The location
                of the ego-vehicle.
        """
        min_index = self.closest_waypoint(location)
        # Remove waypoints that are before the closest waypoint. The ego
        # vehicle already completed them.
        while min_index > 0:
            self.waypoints.popleft()
            if self.target_speeds:
                self.target_speeds.popleft()
            if self.road_options:
                self.road_options.popleft()
            min_index -= 1

        if self.waypoints is None or len(self.waypoints) == 0:
            if ego_transform is not None:
                self.waypoints = deque([ego_transform])
                self.target_speeds = deque([0])
                self.road_options = deque([RoadOption.LANE_FOLLOW])
            else:
                raise ValueError('No more waypoints to follow')

    def recompute_waypoints(self, hd_map, ego_location, goal_location):
        self.waypoints = hd_map.compute_waypoints(ego_location, goal_location)
        self.target_speeds = deque([0 for _ in range(len(self.waypoints))])

    def _get_index(self, transform: Transform,
                   min_distance: float) -> int:
        min_index = -1
        for index, wp in enumerate(self.waypoints):
            distance = wp.location.distance(transform.location)
            if distance >= min_distance:
                min_index = index
                break
        if min_index == -1:
            min_index = len(self.waypoints) - 1
        if min_index < 0:
            raise ValueError('No more waypoints')
        return min_index

    def get_angle(self, transform: Transform,
                  min_distance: float) -> float:
        """Returns the angle between the transform and the first waypoint that
        is at least min_distance away."""
        wp_index = self._get_index(transform, min_distance)
        angle, _ = transform.get_angle_and_magnitude(
            self.waypoints[wp_index].location)
        return angle

    def get_vector(self, transform: Transform,
                   min_distance: float):
        """Returns the vector between the transform and the first waypoint that
        is at least min_distance away."""
        wp_index = self._get_index(transform, min_distance)
        return self.waypoints[wp_index].location.as_vector_2D() - \
            transform.location.as_vector_2D()

    def get_target_speed(self, transform: Transform,
                         min_distance: float) -> float:
        """Gets the target speed at the first waypoint that is at least
        min_distance away."""
        wp_index = self._get_index(transform, min_distance)
        return self.target_speeds[wp_index]

    def slice_waypoints(self,
                        start_index: int,
                        end_index: int,
                        target_speed: float = None):
        head_wps = deque(
            itertools.islice(self.waypoints, start_index, end_index))
        if target_speed is not None:
            # Use the specified target speed.
            head_target_speeds = deque(
                [target_speed for _ in range(len(head_wps))])
        else:
            # Otherwise use the already existing target speeds.
            head_target_speeds = deque(
                itertools.islice(self.target_speeds, start_index, end_index))
        if self.road_options is not None:
            head_road_options = deque(
                itertools.islice(self.road_options, start_index, end_index))
        else:
            head_road_options = None
        return Waypoints(head_wps, head_target_speeds, head_road_options)

    def draw_on_frame(self, bgr_frame, inverse_transform=None):
        """Draw waypoints on a frame.

        Args:
            bgr_frame: Frame on which to draw the waypoints.
            inverse_transform (optional): To be used to transform the waypoints
                to relative to the ego vehicle.
        """
        extrinsic_matrix = bgr_frame.camera_setup.get_extrinsic_matrix()
        intrinsic_matrix = bgr_frame.camera_setup.get_intrinsic_matrix()
        for index, wp in enumerate(self.waypoints):
            if inverse_transform:
                wp = inverse_transform * wp
            pixel_location = wp.location.to_camera_view(
                extrinsic_matrix, intrinsic_matrix)
            bgr_frame.draw_point(pixel_location, [255, 255, 255])
            waypoint_txt = ''
            if self.target_speeds:
                waypoint_txt = '{:.1f}m/s'.format(self.target_speeds[index])
            # if self.road_options:
            #     waypoint_txt = '{} {}'.format(waypoint_txt,
            #                                   self.road_options[index])
            if waypoint_txt != '':
                bgr_frame.draw_text(pixel_location, waypoint_txt,
                                    [255, 255, 255])

    def draw_on_world(self, world):
        """Draw waypoints on the simulator world."""
        for index, wp in enumerate(self.waypoints):
            # Adds 0.5 to z to ensure that the point is above the road surface.
            loc = (wp.location +
                   Location(0, 0, 0.5)).as_simulator_location()
            world.debug.draw_point(loc, size=0.1, life_time=0.1)
            # if self.road_options and index < len(self.road_options):
            #     world.debug.draw_string(loc,
            #                             str(self.road_options[index]),
            #                             life_time=DEFAULT_VIS_TIME)
