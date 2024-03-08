from collections import deque
from objects.objects import Location, Rotation, Transform, Waypoints
from planning.world import World

from frenet_optimal_trajectory_planner.FrenetOptimalTrajectory.fot_wrapper import run_fot

import time
import params

class WaypointPlanner():
    def __init__(self):
        self._world = World()

        if params.planner_type == 'waypoints' or params.planner_type == 'fot':
            self._planner = FOTPlanner()
        elif params.planner_type == 'hybrid':
            self._planner = HybridAStarPlanner()
        else:
            self._planner = RRTStarPlanner()


    def get_waypoints(self, pose, predictions, obstacles, ttd, lanes, type):  
        start = time.time()
        self._world.update(pose, predictions, obstacles, None, None)
        # Total ttd - time spent up to now
        ttd = ttd - (time.time() - self._world.pose.localization_time)

        # if self._state == BehaviorPlannerState.OVERTAKE:
        #     # Ignore traffic lights and obstacle.
        #     output_wps = self._planner.run(timestamp, ttd)
        # else:
        (speed_factor, _, _, speed_factor_tl,
            speed_factor_stop) = self._world.stop_for_agents()
        if type == 'waypoint':
            target_speed = speed_factor * params.target_speed
            output_wps = self._world.follow_waypoints(target_speed)
        else:
            output_wps = self._planner.run(ttd)
            speed_factor = min(speed_factor_stop, speed_factor_tl)
            output_wps.apply_speed_factor(speed_factor)
        
        return output_wps, (time.time() - start) * 1000


class Planner(object):
    """Base planning class.

    All planners implement this interface.
    """
    def __init__(self, world):
        self._world = world
        # TODO: Deal with the map
        self._map = None

    def build_output_waypoints(self, path_x, path_y, speeds):
        """Builds a Waypoints object from 2D locations and speeds."""
        wps = deque()
        target_speeds = deque()
        for point in zip(path_x, path_y, speeds):
            if self._map is not None:
                # Use the HD Map to transform a 2D location into a
                # 3D location.
                p_loc = self._map.get_closest_lane_waypoint(
                    Location(x=point[0], y=point[1], z=0)).location
            else:
                p_loc = Location(x=point[0], y=point[1], z=0)
            # Use the computed x and y (not the ones returned by the HDMap)
            # to ensure that the vehicles follows the computed plan.
            wps.append(
                Transform(
                    location=Location(x=point[0], y=point[1], z=p_loc.z),
                    rotation=Rotation(),
                ))
            target_speeds.append(point[2])
        return Waypoints(wps, target_speeds)


class FOTPlanner(Planner):
    """Frenet Optimal Trajectory (FOT) planner.

    This planner uses a global route and predictions to produce a frenet
    optimal trajectory plan. Details can be found at
    `Frenet Optimal Trajectory Planner`_.

    .. _Frenet Optimal Trajectory Planner:
       https://github.com/erdos-project/frenet_optimal_trajectory_planner
    """
    def __init__(self, world):
        super().__init__(world)
        self.s0 = 0.0
        self._hyperparameters = {
            "num_threads": 1,
            "max_speed": 35.0,
            "max_accel": 6.0,
            "max_curvature": 1.0,
            "max_road_width_l": 5.0,
            "max_road_width_r": 1.0,
            "d_road_w": 0.25,
            "dt": 0.25,
            "maxt": 8.0,
            "mint": 2.0,
            "d_t_s": 0.25,
            "n_s_sample": 2.0,
            "obstacle_clearance": 0.5,
            "kd": 1.0,
            "kv": 0.1,
            "ka": 0.1,
            "kj": 0.01,
            "kt": 0.01,
            "ko": 0.1,
            "klat": 1.0,
            "klon": 1.0
        }

    def fot_parameters_using_99_percentile(self, ttd):
        maxt = 8.0
        runtimes = [309, 208, 148, 67, 40]
        dts = [0.09, 0.11, 0.13, 0.19, 0.31]
        d_road_ws = [0.3, 0.3, 0.3, 0.5, 0.7]

        for index, runtime in enumerate(runtimes):
            if ttd >= runtime:
                return maxt, dts[index], d_road_ws[index]
        # Not enough time to run the planner.
        print('Not enough time to run the planner. Using the fastest version')
        return maxt, dts[-1], d_road_ws[-1]

    def update_hyper_parameters(self, ttd):
        """Changes planning hyper parameters depending on time to decision."""
        # Change hyper paramters if static or dynamic deadlines are enabled.
        if params.deadline_enforcement == 'dynamic':
            maxt, dt, d_road_w = self.fot_parameters_using_99_percentile(ttd)
        elif params.deadline_enforcement == 'static':
            maxt, dt, d_road_w = self.fot_parameters_using_99_percentile(
                params.planning_deadline)
        else:
            return
        self._hyperparameters['maxt'] = maxt
        self._hyperparameters['dt'] = dt
        self._hyperparameters['d_road_w'] = d_road_w

    def run(self, ttd=None):
        """Runs the planner.

        Note:
            The planner assumes that the world is up-to-date.

        Returns:
            :py:class:`~pylot.planning.waypoints.Waypoints`: Waypoints of the
            planned trajectory.
        """
        self.update_hyper_parameters(ttd)
        initial_conditions = self._compute_initial_conditions()
        start = time.time()
        (path_x, path_y, speeds, ix, iy, iyaw, d, s, speeds_x, speeds_y, misc,
         costs, success) = run_fot(initial_conditions, self._hyperparameters)
        fot_runtime = (time.time() - start) * 1000
        if success:
            output_wps = self.build_output_waypoints(path_x, path_y, speeds)
        else:
            output_wps = self._world.follow_waypoints(0)

        # update current pose
        self.s0 = misc['s']
        return output_wps

    def _compute_initial_conditions(self):
        ego_transform = self._world.ego_transform
        obstacle_list = self._world.get_obstacle_list()
        current_index = self._world.waypoints.closest_waypoint(
            ego_transform.location)
        # compute waypoints offset by current location
        wps = self._world.waypoints.slice_waypoints(
            max(current_index - self._flags.num_waypoints_behind, 0),
            min(current_index + self._flags.num_waypoints_ahead,
                len(self._world.waypoints.waypoints)))
        initial_conditions = {
            'ps': self.s0,
            'target_speed': self._flags.target_speed,
            'pos': ego_transform.location.as_numpy_array_2D(),
            'vel': self._world.ego_velocity_vector.as_numpy_array_2D(),
            'wp': wps.as_numpy_array_2D().T,
            'obs': obstacle_list,
        }
        return initial_conditions


class HybridAStarPlanner(Planner):
    """Wrapper around the Hybrid A* planner.

    Note:
        Details can be found at `Hybrid A* Planner`_.

    Args:
        world: (:py:class:`~pylot.planning.world.World`): A reference to the
            planning world.
        flags (absl.flags): Object to be used to access absl flags.

    .. _Hybrid A* Planner:
       https://github.com/erdos-project/hybrid_astar_planner
    """
    def __init__(self, world, flags, logger):
        super().__init__(world, flags, logger)
        self._hyperparameters = {
            "step_size": flags.step_size_hybrid_astar,
            "max_iterations": flags.max_iterations_hybrid_astar,
            "completion_threshold": flags.completion_threshold,
            "angle_completion_threshold": flags.angle_completion_threshold,
            "rad_step": flags.rad_step,
            "rad_upper_range": flags.rad_upper_range,
            "rad_lower_range": flags.rad_lower_range,
            "obstacle_clearance": flags.obstacle_clearance_hybrid_astar,
            "lane_width": flags.lane_width_hybrid_astar,
            "radius": flags.radius,
            "car_length": flags.car_length,
            "car_width": flags.car_width,
        }

    def run(self, timestamp, ttd=None):
        """Runs the planner.

        Note:
            The planner assumes that the world is up-to-date.

        Returns:
            :py:class:`~pylot.planning.waypoints.Waypoints`: Waypoints of the
            planned trajectory.
        """
        obstacle_list = self._world.get_obstacle_list()

        if len(obstacle_list) == 0:
            # Do not use Hybrid A* if there are no obstacles.
            output_wps = self._world.follow_waypoints(self._flags.target_speed)
        else:
            # Hybrid a* does not take into account the driveable region.
            # It constructs search space as a top down, minimum bounding
            # rectangle with padding in each dimension.
            self._logger.debug("@{}: Hyperparameters: {}".format(
                timestamp, self._hyperparameters))
            initial_conditions = self._compute_initial_conditions(
                obstacle_list)
            self._logger.debug("@{}: Initial conditions: {}".format(
                timestamp, initial_conditions))
            path_x, path_y, _, success = apply_hybrid_astar(
                initial_conditions, self._hyperparameters)
            if success:
                self._logger.debug(
                    "@{}: Hybrid A* succeeded".format(timestamp))
                speeds = [self._flags.target_speed] * len(path_x)
                self._logger.debug("@{}: Hybrid A* Path X: {}".format(
                    timestamp, path_x.tolist()))
                self._logger.debug("@{}: Hybrid A* Path Y: {}".format(
                    timestamp, path_y.tolist()))
                self._logger.debug("@{}: Hybrid A* Speeds: {}".format(
                    timestamp, speeds))
                output_wps = self.build_output_waypoints(
                    path_x, path_y, speeds)
            else:
                self._logger.error("@{}: Hybrid A* failed. "
                                   "Sending emergency stop.".format(timestamp))
                output_wps = self._world.follow_waypoints(0)
        return output_wps

    def _compute_initial_conditions(self, obstacles):
        ego_transform = self._world.ego_transform
        start = np.array([
            ego_transform.location.x,
            ego_transform.location.y,
            np.deg2rad(ego_transform.rotation.yaw),
        ])
        self._world.waypoints.remove_completed(ego_transform.location)
        end_index = min(self._flags.num_waypoints_ahead,
                        len(self._world.waypoints.waypoints) - 1)
        if end_index < 0:
            # If no more waypoints left. Then our location is our end wp.
            self._logger.debug("@{}: No more waypoints left")
            end_wp = ego_transform
        else:
            end_wp = self._world.waypoints.waypoints[end_index]
        end = np.array([
            end_wp.location.x, end_wp.location.y,
            np.deg2rad(ego_transform.rotation.yaw)
        ])
        initial_conditions = {
            "start": start,
            "end": end,
            "obs": obstacles,
        }
        return initial_conditions


class RRTStarPlanner(Planner):
    """Wrapper around the RRT* planner.

    Note:
        Details can be found at `RRT* Planner`_.

    Args:
        world: (:py:class:`~pylot.planning.world.World`): A reference to the
            planning world.
        flags (absl.flags): Object to be used to access absl flags.

    .. _RRT* Planner:
       https://github.com/erdos-project/rrt_star_planner
    """
    def __init__(self, world, flags, logger):
        super().__init__(world, flags, logger)
        self._hyperparameters = {
            "step_size": flags.step_size,
            "max_iterations": flags.max_iterations,
            "end_dist_threshold": flags.end_dist_threshold,
            "obstacle_clearance": flags.obstacle_clearance_rrt,
            "lane_width": flags.lane_width,
        }

    def run(self, timestamp, ttd=None):
        """Runs the planner.

        Note:
            The planner assumes that the world is up-to-date.

        Returns:
            :py:class:`~pylot.planning.waypoints.Waypoints`: Waypoints of the
            planned trajectory.
        """
        obstacle_list = self._world.get_obstacle_list()
        if len(obstacle_list) == 0:
            # Do not use RRT* if there are no obstacles.
            # Do not use Hybrid A* if there are no obstacles.
            output_wps = self._world.follow_waypoints(self._flags.target_speed)
        else:
            # RRT* does not take into account the driveable region.
            # It constructs search space as a top down, minimum bounding
            # rectangle with padding in each dimension.
            self._logger.debug("@{}: Hyperparameters: {}".format(
                timestamp, self._hyperparameters))
            initial_conditions = self._compute_initial_conditions(
                obstacle_list)
            self._logger.debug("@{}: Initial conditions: {}".format(
                timestamp, initial_conditions))
            path_x, path_y, success = apply_rrt_star(initial_conditions,
                                                     self._hyperparameters)
            if success:
                self._logger.debug("@{}: RRT* succeeded".format(timestamp))
                speeds = [self._flags.target_speed] * len(path_x)
                self._logger.debug("@{}: RRT* Path X: {}".format(
                    timestamp, path_x.tolist()))
                self._logger.debug("@{}: RRT* Path Y: {}".format(
                    timestamp, path_y.tolist()))
                self._logger.debug("@{}: RRT* Speeds: {}".format(
                    timestamp, speeds))
                output_wps = self.build_output_waypoints(
                    path_x, path_y, speeds)
            else:
                self._logger.error("@{}: RRT* failed. "
                                   "Sending emergency stop.".format(timestamp))
                output_wps = self._world.follow_waypoints(0)
        return output_wps

    def _compute_initial_conditions(self, obstacles):
        ego_transform = self._world.ego_transform
        self._world.waypoints.remove_completed(ego_transform.location)
        end_index = min(self._flags.num_waypoints_ahead,
                        len(self._world.waypoints.waypoints) - 1)
        if end_index < 0:
            # If no more waypoints left. Then our location is our end wp.
            self._logger.debug("@{}: No more waypoints left")
            end_wp = ego_transform
        else:
            end_wp = self._world.waypoints.waypoints[end_index]
        initial_conditions = {
            "start": ego_transform.location.as_numpy_array_2D(),
            "end": end_wp.location.as_numpy_array_2D(),
            "obs": obstacles,
        }
        return initial_conditions


def get_map(host: str = "localhost", port: int = 2000, timeout: int = 10):
    """Get a handle to the world running inside the simulation.

    Args:
        host (:obj:`str`): The host where the simulator is running.
        port (:obj:`int`): The port to connect to at the given host.
        timeout (:obj:`int`): The timeout of the connection (in seconds).

    Returns:
        A tuple of `(client, world)` where the `client` is a connection to the
        simulator and `world` is a handle to the world running inside the
        simulation at the host:port.
    """
    try:
        from carla import Client
        client = Client(host, port)
        client_version = client.get_client_version()
        server_version = client.get_server_version()
        err_msg = 'Simulator client {} does not match server {}'.format(
            client_version, server_version)
        assert client_version == server_version, err_msg
        client.set_timeout(timeout)
    except RuntimeError as r:
        raise Exception("Received an error while connecting to the "
                        "simulator: {}".format(r))
    except ImportError:
        raise Exception('Error importing CARLA.')
    return client.get_world().get_map()