import params
import pygame

from objects.objects import Location

visualize_rgb_camera = True
visualize_detected_obstacles = False
visualize_tracked_obstacles = True
visualize_waypoints = True
visualize_prediction = False
visualize_depth_camera = False
visualize_pose = True

DEFAULT_VIS_TIME = 30000.0

class Visualizer():
    def __init__(self):        
        pygame.init()
        self.display = pygame.display.set_mode(
            (params.camera_image_width, params.camera_image_height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("CARLA WRAPPER")

        # Array of keys to figure out which message to display.
        self.current_display = 0
        self.display_array = []
        self.window_titles = []
        if visualize_rgb_camera:
            self.display_array.append("RGB")
            self.window_titles.append("RGB Camera")
        if visualize_detected_obstacles:
            self.display_array.append("Obstacle")
            self.window_titles.append("Detected obstacles")
        if visualize_tracked_obstacles:
            self.display_array.append("TrackedObstacle")
            self.window_titles.append("Obstacle tracking")
        if visualize_waypoints:
            self.display_array.append("Waypoint")
            self.window_titles.append("Planning")
        if visualize_prediction:
            self.display_array.append("PredictionCamera")
            self.window_titles.append("Prediction")
        if visualize_depth_camera:
            self.display_array.append("Depth")
            self.window_titles.append("Depth Camera")
        
    def visualize(self, timestamp, frame, depth_frame, pose, obstacles, throttle, steer, brake):
        ego_transform = pose.transform
        self.visualize_pose(ego_transform)
        frame.visualize(self.display, timestamp=timestamp)
        depth_frame.visualize(self.display, timestamp=timestamp)
        frame.annotate_with_bounding_boxes(timestamp, obstacles, ego_transform)
        self.render_text(pose, throttle, steer, brake, timestamp)
    
    def render_text(self, pose, throttle, steer, brake, timestamp):
        # Generate the text to be shown on the box.
        info_text = [
            "Display  : {}".format(self.window_titles[self.current_display]),
            "Timestamp: {}".format(timestamp),
        ]

        # Add information from the pose.
        if pose:
            info_text += [
                "Location : {:.1f}, {:.1f}, {:.1f}".format(
                    *tuple(pose.transform.location.as_numpy_array())),
                "Rotation : {:.1f}, {:.1f}, {:.1f}".format(
                    *tuple(pose.transform.rotation.as_numpy_array())),
                "Speed    : {:.2f} m/s".format(pose.forward_speed),
            ]

        # Add information from the control message
        info_text += [
            "Throttle : {:.2f}".format(throttle),
            "Steer    : {:.2f}".format(steer),
            "Brake    : {:.2f}".format(brake),
        ]

        # Display the information box.
        info_surface = pygame.Surface((220, self._flags.camera_image_height // 3))
        info_surface.set_alpha(100)
        self.display.blit(info_surface, (0, 0))

        # Render the text.
        v_offset = 10
        for line in info_text:
            if v_offset + 18 > params.camera_image_height:
                break
            surface = self.font.render(line, True, (255, 255, 255))
            self.display.blit(surface, (8, v_offset))
            v_offset += 18
        pygame.display.flip()
    
    def visualize_pose(self, ego_transform):
        loc = (ego_transform.location + Location(0, 0, 0.5)).as_simulator_location()
        self._world.debug.draw_point(loc, size=0.2, life_time=DEFAULT_VIS_TIME)
