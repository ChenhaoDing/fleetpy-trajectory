import pandas as pd
import numpy as np
import os
import logging

from src.NetworkRouter import NetworkRouter


class VehTrajBuilder:
    def __init__(self, network_router: NetworkRouter, logger: logging.Logger = None):
        if logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            # Avoid adding handler if logger already has handlers (e.g., in Jupyter)
            if not self.logger.hasHandlers():
                self.logger.addHandler(logging.StreamHandler())
        else: 
            self.logger = logger

        self.network_router: NetworkRouter = network_router

    def _query_travel_time_1to1(self, start_node_id: int, end_node_id: int) -> float:
        _, _, travel_time = self.network_router.get_shortest_path_details(start_node_id, end_node_id, 'travel_time')
        return travel_time

    def build_vehicle_timeline(self, veh_stats: pd.DataFrame, veh_id: int)-> dict:
        timeline: dict = {}  # Initialize an empty dictionary to store the timeline: {time: [node_id, occupancy]}
        record_count: int = 0

        try:
            # only keep rows where status is 'route'
            veh_stats: pd.DataFrame = veh_stats[veh_stats['status'] == 'route']

            # sort the dataframe by start_time
            veh_stats: pd.DataFrame = veh_stats.sort_values(by='start_time')

            for _, row in veh_stats.iterrows():
                record_count += 1
                start_time: float = float(row['start_time'])
                end_time: float = row['end_time']
                occupancy = int(row['occupancy'])

                route_str: str = row['route']
                route: str = route_str.strip().split(';')
                route: list[int] = [int(node_id) for node_id in route]

                if not route:
                    self.logger.warning(f"Warning: The {record_count} record has an empty route: '{route_str}', skipping.")
                    continue

                timeline[start_time] = (route[0], occupancy)

                actual_duration = end_time - start_time
                if actual_duration < 0:
                    self.logger.warning(f"Warning: The {record_count} record has an end time earlier than the start time, skipping.")
                    continue

                # process route with one or two nodes
                if len(route) <= 1:
                    if start_time != end_time:
                        timeline[end_time] = (route[0], occupancy)
                    continue
                if len(route) == 2:
                    timeline[end_time] = (route[1], occupancy)
                    continue

                # vehicle does not move
                if actual_duration == 0 and len(route) > 2:
                    current_time = start_time
                    for i in range(2, len(route)):
                        timeline[current_time] = (route[i], occupancy)
                    continue

                total_estimated_time = 0
                segment_times = []
                valid_segment = True
                for i in range(len(route) - 1):
                    segment_time = self._query_travel_time_1to1(route[i], route[i+1])
                    if segment_time < 0:
                        self.logger.warning(f"Warning: The returned travel time for {route[i]}->{route[i+1]} is negative: ({segment_time}). Skipping this record ({record_count}).")
                        valid_segment = False
                        break
                    segment_times.append(segment_time)
                    total_estimated_time += segment_time
                
                if not valid_segment:
                        continue
                
                current_time = start_time
                if total_estimated_time > 0:
                    for i in range(len(route) - 2):
                        estimated_sub_segment_time = segment_times[i]
                        actual_sub_segment_time = actual_duration * (estimated_sub_segment_time / total_estimated_time)
                        current_time += actual_sub_segment_time
                        intermediate_point = route[i+1]
                        timeline[current_time] = (intermediate_point, occupancy)
                elif len(route) > 1:
                        intermediate_time = start_time
                        for i in range(1, len(route) - 1):
                            timeline[intermediate_time] = (route[i], occupancy)
                
                timeline[end_time] = (route[-1], 0)

            self.logger.info(f"The Vehicle Trajectory for vehicle {veh_id} has been built.")
            return timeline

        except Exception as e:
            self.logger.error(f"Unexpected error building vehicle trajectory: {e}")
            return None

    def build_fleet_trajectory(
            self, 
            fleet_stats: dict[int, pd.DataFrame],
            start_time: int,
            end_time: int,
            time_step: int = 1,
            ) -> dict:
        fleet_trajectory: dict = {}  # Initialize an empty dictionary to store the fleet timeline: {vehicle_id: {time: [node_id, occupancy]}}

        time_steps: np.ndarray = np.arange(start_time, end_time+1, time_step)

        for veh_id, veh_stats in fleet_stats.items():
            veh_timeline = self.build_vehicle_timeline(veh_stats, veh_id)
            if veh_timeline is None:
                continue
            veh_trajectory: dict = {}  # Initialize an empty dictionary to store the vehicle trajectory: {time: [node_id, occupancy]}

            veh_time_steps = sorted(veh_timeline.keys())

            for current_time_step in time_steps:
                # find the closest time in veh_time_steps to the current time_step
                closest_time = min(veh_time_steps, key=lambda x: abs(x - current_time_step))
                veh_trajectory[current_time_step] = veh_timeline[closest_time]

            fleet_trajectory[veh_id] = veh_trajectory

        self.logger.info(f"The Fleet Trajectory from {start_time} to {end_time} with a time step of {time_step} has been built.")
        self.logger.info(f"The Fleet Trajectory contains {len(fleet_trajectory.keys())} vehicles.")
        return fleet_trajectory

