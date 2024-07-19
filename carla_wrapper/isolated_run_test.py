import pandas as pd
import time
from control.controller import Controller

class MPCRunner():

    def __init__(self):
        self._controller = Controller()
        self.throttle = -1
        self.brake = -1
        self.steer = -1

    def run_MPC(self, timestamp, pose, waypoints):
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

def main():
    # Load data from the CSV file
    file_path = 'planner_dump.csv'
    data = pd.read_csv(file_path)
    
    runner = MPCRunner()
    
    for index, row in data.iterrows():
        timestamp = row['timestamp']
        pose = eval(row['pose'])  # Convert string representation back to original object
        waypoints = eval(row['waypoints'])  # Convert string representation back to original object
        
        runner.run_MPC(timestamp, pose, waypoints)

if __name__ == '__main__':
    main()