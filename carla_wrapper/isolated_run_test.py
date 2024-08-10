import csv
import pandas as pd
import time
import numpy as np
from collections import deque
from control.controller import Controller

class Vector3D(object):
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self.x = x
        self.y = y
        self.z = z

class Location(Vector3D):
    '''
    XYZ Position of Vehicle
    '''
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        super().__init__(x, y, z)

    def __str__(self):
        return "(Location : "+str(self.x)+", "+str(self.y)+", "+str(self.z)+")"

class Rotation(object):
    '''
    Orientation (Roll, Pitch, Yaw)
    '''
    def __init__(self, pitch: float = 0, yaw: float = 0, roll: float = 0):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll
    
    def __str__(self):
        return "(Rotation : "+str(self.pitch)+", "+str(self.yaw)+", "+str(self.roll)+")"

class Transform(object):
    '''
    Position and Orientation of vehicle
    '''
    def __init__(self, location: Location = None, rotation: Rotation = None, matrix=None):
        self.location = location
        self.rotation = rotation
        self.matrix = matrix
    
    def __str__(self) -> str:
        return "("+str(self.location)+", "+str(self.rotation)+")"

class Pose(object):
    '''
    Stores state of vehicle to be controlled. Contains vehicle transform (position and orientation) and velocity
    '''
    def __init__(self, transform: Transform, forward_speed: float, velocity_vector: Vector3D = None, localization_time: float = None):
        self.transform = transform
        self.forward_speed = forward_speed
        self.velocity_vector = velocity_vector
        self.localization_time = localization_time
    
    def __str__(self) -> str:
        return "("+str(self.transform)+", Speed : "+str(self.forward_speed)+")"

class Waypoints(object):
    '''
    Stores a sequence of waypoints and target speeds. Each waypoint
    Args:
        waypoints (deque<Transform>) - deque of transform objects. (Target configurations)
        target_speeds (deque<float>) - deque of target speeds
    '''
    def __init__(self, waypoints: deque, target_speeds: deque):
        self.waypoints = waypoints
        self.target_speeds = target_speeds

    # @classmethod
    # def read_from_csv_file(cls, csv_file_name: str, target_speed):
    #     csv_file = open(csv_file_name)
    #     csv_reader = csv.reader(csv_file)
    #     waypoints = []
    #     for row in csv_reader:
    #         x = float(row[0])
    #         y = float(row[1])
    #         z = float(row[2])
    #         waypoint = Transform(Location(x, y, z), Rotation(0, 0, 0))
    #         waypoints.append(waypoint)
    #     target_speeds = deque([target_speed for _ in range(len(waypoints))])
    #     return cls(deque(waypoints), target_speeds)

    def as_numpy_array_2D(self):
        '''
        Returns only x and y positions as numpy array
        '''
        wx = []
        wy =[]
        for wp in self.waypoints:
            wx.append(wp.location.x)
            wy.append(wp.location.y)
        return np.array([wx,wy])
    
    def __str__(self) -> str:
        return str([(str(self.waypoints[i]), str(self.target_speeds[i])) for i,wp in enumerate(self.waypoints)])

class MPCRunner():

    def __init__(self):
        self._controller = Controller()
        self.throttle = -1
        self.brake = -1
        self.steer = -1

    def run_MPC(self, timestamp, pose : Pose, waypoints : Waypoints):
        start_time = time.time()
        (steer, throttle, brake, controller_runtime) = self._controller.get_control_instructions(timestamp, pose, waypoints)
        end_time = time.time()
        print("\nControl instructions {} {} {} {}".format(throttle, steer, brake, controller_runtime))
        print("Calculated runtime: ", end_time - start_time)
        print("Forwarded runtime: ", controller_runtime)

        if throttle == 0 and brake == 0.5 and steer == 0:
            if self.throttle != -1 and self.brake != -1:
                throttle = self.throttle
                brake = self.brake
                steer = self.steer

        self.throttle = throttle
        self.brake = brake
        self.steer = steer

def parse_pose(pose_str):
    transform_str = pose_str.split('transform: ')[1].split('), forward_speed')[0]
    forward_speed = float(pose_str.split('forward speed: ')[1].split(',')[0])
    location_str = transform_str.split('location: Location(')[1].split('), rotation:')[0]
    rotation_str = transform_str.split('rotation: Rotation(')[1].split(')')[0]

    x, y, z = [float(value.split('=')[1]) for value in location_str.split(', ')]
    pitch, yaw, roll = [float(value.split('=')[1]) for value in rotation_str.split(', ')]

    location = Location(x, y, z)
    rotation = Rotation(pitch, yaw, roll)
    transform = Transform(location, rotation)

    return Pose(transform, forward_speed)

def parse_waypoints(waypoints_str):
    waypoints_list = waypoints_str[6:-1].split('), Transform')
    waypoints = []
    for waypoint_str in waypoints_list:
        waypoint_str = 'Transform' + waypoint_str if not waypoint_str.startswith('Transform') else waypoint_str
        location_str = waypoint_str.split('location: Location(')[1].split('), rotation:')[0]
        rotation_str = waypoint_str.split('rotation: Rotation(')[1].split(')')[0]

        x, y, z = [float(value.split('=')[1]) for value in location_str.split(', ')]
        pitch, yaw, roll = [float(value.split('=')[1]) for value in rotation_str.split(', ')]

        location = Location(x, y, z)
        rotation = Rotation(pitch, yaw, roll)
        transform = Transform(location, rotation)

        waypoints.append(transform)

    target_speeds = deque([0] * len(waypoints))
    return Waypoints(deque(waypoints), target_speeds)

def main():
    file_path = "G:\\UCLA\\Research\\IoBT\\pylot\\carla_wrapper\\planner_dump.csv"
    data = pd.read_csv(file_path)
    
    runner = MPCRunner()
    
    for index, row in data.iterrows():
        timestamp = row['timestamp']
        pose = parse_pose(row['pose'])
        waypoints = parse_waypoints(row['waypoints']) # Target speed is always 0?
        
        runner.run_MPC(timestamp, pose, waypoints)

if __name__ == '__main__':
    main()
