# Simulator params

simulator_host = '127.0.0.1'
simulator_port = 2000
simulator_timeout = 10

simulator_camera_frequency = -1

camera_image_width = 1024
camera_image_height = 576
camera_fov = 90.0

# Service params

host="0.0.0.0"
port=5010

device='cpu' # used by torch for r2p2 computation

# Taken from control/flags.py

simulator_control_frequency = -1
simulator_fps = 20

min_pid_steer_waypoint_distance = 5
min_pid_speed_waypoint_distance = 5
steer_gain = False

stop_for_traffic_lights = True
stop_for_people = True
stop_for_vehicles = True
stop_at_uncontrolled_junctions = False

traffic_light_min_distance = 5
traffic_light_max_distance = 20
traffic_light_max_angle = 0.6
vehicle_max_distance = 18
vehicle_max_angle = 0.4
person_distance_hit_zone = 35
person_angle_hit_zone = 0.15
person_distance_emergency_zone = 15
person_angle_emergency_zone = 0.5

throttle_max = 10.0
steer_gain = 0.7
brake_max = 10.0
coast_factor = 1.75

# Taken from prediction/flags.py

prediction_radius = 50
prediction_num_past_steps = 5
prediction_num_future_steps = 10
prediction_ego_agent = False
r2p2_model_path = '/home/erdos/workspace/pylot/dependencies/models/prediction/r2p2/r2p2-model.pt'

# Taken from planning/flags.py
target_speed = 10.0
obstacle_radius = 1.0
num_waypoints_ahead = 60
num_waypoints_behind = 30
obstacle_filtering_distance = 1.0

# Taken from perception/flags.py

static_obstacle_distance_threshold = 70.0
dynamic_obstacle_distance_threshold = 50.0

traffic_light_det_min_score_threshold = 0.3

tracking_num_steps = 10

min_matching_iou=0.5
obstacle_track_max_age=3
deep_sort_tracker_weights_path='/home/erdos/workspace/pylot/dependencies/models/tracking/deep-sort-carla/feature_extractor'
