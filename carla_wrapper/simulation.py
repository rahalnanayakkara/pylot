import time

import params

from utils.simulation import set_mode_fps
import utils.logging

from carla import VehicleControl, command
import carla

from objects.objects import Obstacle, TrafficLight, StopSign, SpeedLimitSign
from objects.objects import Transform, Vector3D, Pose, Rotation, Location
from objects.objects import RGBCameraSetup, DepthCameraSetup
from camera.carla_camera import CarlaCamera

simulator_host = 'localhost'
simulator_port = 2000
simulator_timeout = 10
carla_traffic_manager_port = 8000

# The location of the center camera relative to the ego-vehicle.
CENTER_CAMERA_LOCATION = Location(1.3, 0.0, 1.8)

transform = Transform(CENTER_CAMERA_LOCATION, Rotation(pitch=-15))

rgb_camera_setup = RGBCameraSetup('center_camera',
                                    params.camera_image_width,
                                    params.camera_image_height, transform,
                                    params.camera_fov)

depth_camera_setup = DepthCameraSetup('depth_center_camera',
                                    params.camera_image_width,
                                    params.camera_image_height, transform,
                                    params.camera_fov)

class CarlaSimulation:

    def __init__(self, client, world):
        # Dump logs for CarlaSimulation
        self._module_logger = utils.logging.get_module_logger("CarlaSimulation")

        print("\nInitializing world ...")
        self._client = client
        self._world = world
        set_mode_fps(self._world, params.simulator_fps)

        self._game_time = 0
        self._spectator = self._world.get_spectator()
        
        print("\nTick until 4.0 seconds time so all runs are synchronized ...")
        self.tick_simulator_until(4000)

        print("\nWaiting for ego vehicle to spawn ...")
        self._ego_vehicle = self.wait_for_ego_vehicle(self._world)

        print("\nAdding cameras to vehicle ...")
        self._camera = CarlaCamera(self._world, self._ego_vehicle, rgb_camera_setup)
        self._depth_camera = CarlaCamera(self._world, self._ego_vehicle, depth_camera_setup)

        print("\nAdding ontick callback")
        self._world.on_tick(self.on_simulator_tick)

    def tick_simulator_until(self, goal_time: int):
        while True:
            snapshot = self._world.get_snapshot()
            sim_time = int(snapshot.timestamp.elapsed_seconds * 1000)
            if sim_time < goal_time:
                self._world.tick()
            else:
                return
    
    def wait_for_ego_vehicle(self, world):
        # Connect to the ego-vehicle spawned by the scenario runner.
        while True:
            time.sleep(0.1)
            possible_actors = world.get_actors().filter('vehicle.*')
            for actor in possible_actors:
                if actor.attributes['role_name'] == 'hero':
                    return actor
            world.tick()

    def apply_control(self, throttle, steer, brake, hand_brake, reverse):
        vec_control = VehicleControl(throttle=throttle,
                                     steer=steer,
                                     brake=brake,
                                     hand_brake=hand_brake,
                                     reverse=reverse)
        self._client.apply_batch_sync([command.ApplyVehicleControl(self._ego_vehicle.id, vec_control)])
    
    def tick_simulator(self):
        print("\nTicking simulator ...")
        self._module_logger.info("\nForce ticking simulator ...")
        self._world.tick()
        frame = self._camera.get_processed_image(self._game_time)
        depth_frame = self._depth_camera.get_processed_image(self._game_time)
        pose = self.read_ego_vehicle_data()
        return self._game_time, frame, depth_frame, pose

    def on_simulator_tick(self, msg):
        self._game_time = int(msg.elapsed_seconds * 1000)
        self._module_logger.info("\nWorld is ticking ... " + str(self._game_time))
        self.update_spectator_pose()
    
    def read_ego_vehicle_data(self):
        vec_transform = Transform.from_simulator_transform(self._ego_vehicle.get_transform())
        velocity_vector = Vector3D.from_simulator_vector(self._ego_vehicle.get_velocity())
        forward_speed = velocity_vector.magnitude()
        return Pose(vec_transform, forward_speed, velocity_vector)
    
    def read_ground_actors_data(self):
        actor_list = self._world.get_actors()

        vec_actors = actor_list.filter('vehicle.*')
        vehicles = [Obstacle.from_simulator_actor(vec_actor) for vec_actor in vec_actors]
        
        person_actors = actor_list.filter('walker.pedestrian.*')
        people = [Obstacle.from_simulator_actor(ped_actor) for ped_actor in person_actors]
        
        tl_actors = actor_list.filter('traffic.traffic_light*')
        traffic_lights = [TrafficLight.from_simulator_actor(tl_actor) for tl_actor in tl_actors]
        
        speed_limit_actors = actor_list.filter('traffic.speed_limit*')
        speed_limits = [SpeedLimitSign.from_simulator_actor(ts_actor) for ts_actor in speed_limit_actors]
        
        traffic_stop_actors = actor_list.filter('traffic.stop')
        traffic_stops = [StopSign.from_simulator_actor(ts_actor) for ts_actor in traffic_stop_actors]

        return (vehicles, people, traffic_lights, speed_limits, traffic_stops)
    
    def update_spectator_pose(self):
         # Set the world simulation view with respect to the vehicle.
        v_pose = self._ego_vehicle.get_transform()
        v_pose.location -= 10 * carla.Location(v_pose.get_forward_vector())
        v_pose.location.z = 5
        self._spectator.set_transform(v_pose)