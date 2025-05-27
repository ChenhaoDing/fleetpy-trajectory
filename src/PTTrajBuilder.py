from shapely.geometry import Point, LineString, MultiLineString
import geopandas as gpd

import math
import bisect

def build_pt_trajectory(
    route: MultiLineString,
    trajectory_start_time: int,
    trajectory_end_time: int,
    time_step: int = 10,
    reverse: bool = False
):
    # get points
    path_coordinates = []
    for single_linestring in route.geoms:
        for point_tuple in list(single_linestring.coords):
            path_coordinates.append(tuple(point_tuple))

    if reverse:
        path_coordinates.reverse()

    segment_lengths = []
    cumulative_distances = [0.0]
    current_total_distance = 0.0

    for i in range(len(path_coordinates) - 1):
        p1 = path_coordinates[i]
        p2 = path_coordinates[i+1]

        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        segment_lengths.append(dist)
        current_total_distance += dist
        cumulative_distances.append(current_total_distance)

    total_trajectory_length = current_total_distance

    trajectory_total_duration = trajectory_end_time - trajectory_start_time
    velocity = total_trajectory_length / trajectory_total_duration

    times_to_check = []
    current_t_gen = trajectory_start_time
    while current_t_gen <= trajectory_end_time + 1e-9:
        times_to_check.append(current_t_gen)
        current_t_gen += time_step

    output_points = {}
    for t_query in times_to_check:
        if not (trajectory_start_time - 1e-9 <= t_query <= trajectory_end_time + 1e-9):
            continue
        time_on_trajectory = t_query - trajectory_start_time
        
        distance_to_find = velocity * time_on_trajectory
        distance_to_find = max(0.0, min(distance_to_find, total_trajectory_length))

        interpolated_point = None
        if math.isclose(distance_to_find, 0.0):
                interpolated_point = path_coordinates[0]
        elif math.isclose(distance_to_find, total_trajectory_length):
            interpolated_point = path_coordinates[-1]
        else:
            segment_insertion_idx = bisect.bisect_left(cumulative_distances, distance_to_find)
            segment_idx_in_arrays = segment_insertion_idx - 1

            if segment_idx_in_arrays < 0: segment_idx_in_arrays = 0
            if segment_idx_in_arrays >= len(segment_lengths): segment_idx_in_arrays = len(segment_lengths) - 1

            p_start_segment = path_coordinates[segment_idx_in_arrays]
            p_end_segment = path_coordinates[segment_idx_in_arrays + 1]
            length_of_this_segment = segment_lengths[segment_idx_in_arrays]

            if math.isclose(length_of_this_segment, 0.0):
                interpolated_point = p_start_segment
            else:
                dist_into_segment = distance_to_find - cumulative_distances[segment_idx_in_arrays]
                ratio = dist_into_segment / length_of_this_segment
                ratio = max(0.0, min(1.0, ratio))

                interp_x = p_start_segment[0] + ratio * (p_end_segment[0] - p_start_segment[0])
                interp_y = p_start_segment[1] + ratio * (p_end_segment[1] - p_start_segment[1])
                interpolated_point = (interp_x, interp_y)
        if interpolated_point:
                output_points[t_query] = interpolated_point
    return output_points


def build_pt_trajectory_during_study(
    route: MultiLineString,
    study_start_time: int,
    study_end_time: int,
    pt_headway: int,
    pt_duration: int,
    time_step: int = 10,
    reverse: bool = False
):
    trajectory_times_to_check = []
    # get all possible trajectory_start_time, trajectory_end_time
    for t in range(study_start_time, study_end_time, pt_headway):
        if t + pt_duration > study_end_time:
            break
        trajectory_times_to_check.append((t, t + pt_duration))

    pt_trajectories = {}
    for trajectory_start_time, trajectory_end_time in trajectory_times_to_check:
        pt_trajectories[(trajectory_start_time, trajectory_end_time)] = build_pt_trajectory(route, trajectory_start_time, trajectory_end_time, time_step, reverse)
    
    new_pt_trajectories = {}
    for _, nested_dict in pt_trajectories.items():
        for time_key, coordinates in nested_dict.items():
            if time_key not in new_pt_trajectories:
                new_pt_trajectories[time_key] = []
            new_pt_trajectories[time_key].append(coordinates)
    return new_pt_trajectories