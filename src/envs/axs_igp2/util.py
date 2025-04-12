"""Utility functions for the IGP2 AXSAgent implementation."""

from collections import defaultdict

import igp2 as ip
import numpy as np
import numpy.ma as ma


def subsample_trajectory(
    trajectory: ip.StateTrajectory,
    start_t: int,
    f_subsample: int,
) -> ip.StateTrajectory:
    """Subsample a trajectory by a factor of f_subsample.

    Args:
        trajectory: The trajectory to subsample.
        start_t: The start time of the state sequence.
        f_subsample: The factor to subsample by.

    """
    ts = trajectory.times
    states = []

    idx = np.isclose(trajectory.velocity, 0.0)
    nonstops = ma.array(trajectory.velocity, mask=~idx)
    for slicer in ma.clump_masked(nonstops):
        num_frames = (slicer.stop - slicer.start + 1) // f_subsample
        ts_points = ts[slicer]
        points = np.linspace(
            ts_points[0], ts_points[-1], num_frames,
        )
        xs_r = np.interp(points, ts, trajectory.path[:, 0])
        ys_r = np.interp(points, ts, trajectory.path[:, 1])
        v_r = np.interp(points, ts, trajectory.velocity)
        a_r = np.interp(points, ts, trajectory.acceleration)
        h_r = np.interp(points, ts, trajectory.heading)
        path = np.c_[xs_r, ys_r]

        new_states = [
            ip.AgentState(
                time=start_t + slicer.start + i * f_subsample,
                position=path[i],
                velocity=v_r[i],
                acceleration=a_r[i],
                heading=h_r[i],
            )
            for i in range(num_frames)
        ]
        states.append((slicer.start, new_states))

    stops = ma.array(trajectory.velocity, mask=idx)
    for slicer in ma.clump_masked(stops):
        start = int(slicer.start)
        new_states = [
            ip.AgentState(
                time=start_t + start + i * f_subsample,
                position=trajectory.path[start],
                velocity=np.array([0.0, 0.0]),
                acceleration=np.array([0.0, 0.0]),
                heading=trajectory.heading[start],
            )
            for i in range((slicer.stop - slicer.start + 1) // f_subsample)
        ]
        states.append((start, new_states))

    states = sorted(states, key=lambda x: x[0])
    states = [s for _, state in states for s in state]

    fps = None
    if trajectory.fps is not None:
        fps = trajectory.fps // f_subsample
    return ip.StateTrajectory(fps, states=states)


def ndarray2str(array: np.ndarray, precision: int = 2) -> str:
    """Format a numpy array to a string.

    Args:
        array (np.ndarray): The array to format.
        precision (int): The number of decimal places to use.

    """
    if precision < 1:
        error_msg = "Precision must be at least 1."
        raise ValueError(error_msg)

    ret = np.array2string(
        array,
        separator=", ",
        precision=precision,
        suppress_small=True,
    )
    ret = ret.replace("\n", "")
    return " ".join(ret.split())


def infos2traj(
    infos: list[dict[str, int]], time: int | None = None, fps: int | None = None,
) -> dict[int, ip.StateTrajectory]:
    """Convert a list of info dicts to a dictionary of StateTrajectories.

    Args:
        infos: The list of info dicts to convert.
        time: The cut-off time to stop at, truncating the info dict list.
        fps: The frames per second of the trajectories.

    """
    trajectories = defaultdict(list)
    for t, info_dict in enumerate(infos):
        if time is not None and t > time:
            break
        for agent_id, agent_state in info_dict.items():
            trajectories[agent_id].append(agent_state)
    return {
        agent_id: ip.StateTrajectory(fps, trajectory)
        for agent_id, trajectory in trajectories.items()
    }
