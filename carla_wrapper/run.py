import params

from simulation import CarlaSimulation

def main():
    simulation = CarlaSimulation()
    for i in range(1000):
        simulation.tick_simulator()
        simulation.apply_control(1, 0, 0, 0, 0)
        pose = simulation.read_ego_vehicle_data()
        print(str(pose))

if __name__ == '__main__':
    main()