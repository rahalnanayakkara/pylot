import time

import params

from utils.simulation import get_world, set_mode_fps

from carla import VehicleControl, command

from objects.objects import Obstacle, TrafficLight, StopSign, SpeedLimitSign
from objects.objects import Transform, Location, Vector3D, Pose 
from camera.carla_camera import CarlaCamera

simulator_host = 'localhost'
simulator_port = 2000
simulator_timeout = 10
carla_traffic_manager_port = 8000

class CarlaSimulation:

    def __init__(self) -> None:
        print("\nInitializing world ...")
        self._client, self._world = get_world(params.simulator_host,
                             params.simulator_port,
                             params.simulator_timeout)
        set_mode_fps(self._world, params.simulator_fps)

        self._spectator = self._world.get_spectator()
        
        print("\nTick until 4.0 seconds time so all runs are synchronized ...")
        self.tick_simulator_until(4000)

        print("\nWaiting for ego vehicle to spawn ...")
        self._ego_vehicle = self.wait_for_ego_vehicle(self._world)

        print("\nAdding camera to vehicle ...")
        self._camera = CarlaCamera(self._world, self._ego_vehicle.id)

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
        vec_control = VehicleControl(throttle, steer, brake, hand_brake, reverse)
        self._client.apply_batch_sync([command.ApplyVehicleControl(self._ego_vehicle.id, vec_control)])
    
    def tick_simulator(self):
        print("\nForce ticking simulator ...")
        self._world.tick()
    
    def on_simulator_tick(self, msg):
        game_time = int(msg.elapsed_seconds * 1000)
        print("\nWorld is ticking ... " + str(game_time))
        (vehicles, people, traffic, limits, stops) = self.read_ground_actors_data()
        self.update_spectator_pose()
        print(str(vehicles))
    
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
        v_pose.location -= 10 * Location(v_pose.get_forward_vector())
        v_pose.location.z = 5
        self._spectator.set_transform(v_pose)