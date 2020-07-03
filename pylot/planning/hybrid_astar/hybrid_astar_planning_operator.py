"""
Author: Edward Fang
Email: edward.fang@berkeley.edu
"""
import erdos

import numpy as np

from pylot.planning.messages import WaypointsMessage
from pylot.planning.planning_operator import PlanningOperator
from pylot.planning.hybrid_astar.hybrid_astar_planner.HybridAStar.hybrid_astar_wrapper \
    import apply_hybrid_astar


class HybridAStarPlanningOperator(PlanningOperator):
    """HybridAStar Planning operator.

    Args:
        flags: Config flags.
        goal_location: Goal pylot.utils.Location for planner to route to.
    """
    def __init__(self,
                 pose_stream,
                 prediction_stream,
                 static_obstacles_stream,
                 lanes_stream,
                 global_trajectory_stream,
                 open_drive_stream,
                 time_to_decision_stream,
                 waypoints_stream,
                 flags,
                 goal_location=None):
        super().__init__(pose_stream, prediction_stream,
                         static_obstacles_stream, lanes_stream,
                         global_trajectory_stream, open_drive_stream,
                         time_to_decision_stream, waypoints_stream, flags,
                         goal_location)
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

    @erdos.profile_method()
    def on_watermark(self, timestamp, waypoints_stream):
        self._logger.debug('@{}: received watermark'.format(timestamp))
        self.update_world(timestamp)
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
                output_wps = self.follow_waypoints(0)

        waypoints_stream.send(WaypointsMessage(timestamp, output_wps))

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
