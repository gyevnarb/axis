"""Utility functions for the IGP2 AXSAgent implementation."""

from collections import defaultdict
import igp2 as ip
import numpy as np


def subsample_trajectory(
    trajectory: ip.StateTrajectory,
    f_subsample: int,
) -> ip.StateTrajectory:
    """Subsample a trajectory by a factor of f_subsample.

    Args:
        trajectory: The trajectory to subsample.
        f_subsample: The factor to subsample by.

    """
    ts = trajectory.times
    num_frames = np.ceil(len(ts) / f_subsample).astype(int)
    points = np.linspace(ts[0], ts[-1], num_frames)

    xs_r = np.interp(points, ts, trajectory.path[:, 0])
    ys_r = np.interp(points, ts, trajectory.path[:, 1])
    v_r = np.interp(points, ts, trajectory.velocity)
    a_r = np.interp(points, ts, trajectory.acceleration)
    h_r = np.interp(points, ts, trajectory.heading)
    path = np.c_[xs_r, ys_r]

    states = [
        ip.AgentState(
            time=i * f_subsample,
            position=path[i],
            velocity=v_r[i],
            acceleration=a_r[i],
            heading=h_r[i],
        )
        for i in range(num_frames)
    ]

    fps = None
    if trajectory.fps is not None:
        fps = trajectory.fps // f_subsample
    return ip.StateTrajectory(
        fps,
        states=states,
        path=path,
        velocity=v_r,
    )


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
