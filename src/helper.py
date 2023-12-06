import os
import robomimic.utils.file_utils as FileUtils
from robomimic import DATASET_REGISTRY
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def download_dataset(task: str, dataset_type: str, hdf5_type: str, download_dir: str):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    dataset_path = os.path.join(download_dir, "low_dim_v141.hdf5")
    if os.path.exists(dataset_path):
        return dataset_path
    FileUtils.download_url(
    url=DATASET_REGISTRY[task][dataset_type][hdf5_type]["url"],
    download_dir=download_dir,
    )
    # enforce that the dataset exists
    assert os.path.exists(dataset_path)
    return dataset_path

def extract_trajectory_i(hdf5_object, trajectory_idx):
    """
    Prints information about a trajectory in the HDF5 file
    and extracts the states, goal state, and actions for the desired trajectory.

    Args:
        hdf5_object: h5py.File or h5py.Group object
        trajectory_idx (int): index of the trajectory to print
    Returns:
        states (np.ndarray): array of states
        goal_state (np.ndarray): goal state
        actions (np.ndarray): array of actions
    """
    print(f"Extracting Trajectory {trajectory_idx}:")
    
    # Actions
    actions = np.array(hdf5_object['data'][f'demo_{trajectory_idx}']['actions'])
    print(f"    - Actions: {actions.shape}")
    
    # states
    end_effector_pos = hdf5_object['data'][f'demo_{trajectory_idx}']['obs']['robot0_eef_pos']
    end_effector_rot = hdf5_object['data'][f'demo_{trajectory_idx}']['obs']['robot0_eef_quat']
    gripper_joint_pos = hdf5_object['data'][f'demo_{trajectory_idx}']['obs']['robot0_gripper_qpos']
    object = hdf5_object['data'][f'demo_{trajectory_idx}']['obs']['object']
    states = np.concatenate([end_effector_pos, end_effector_rot, gripper_joint_pos, object], axis=-1)
    print(f"    - States: {states.shape}")
    
    # goal state
    goal_state_idx = states.shape[0] - 1
    goal_end_effector_pos = hdf5_object['data'][f'demo_{trajectory_idx}']['next_obs']['robot0_eef_pos'][goal_state_idx]
    goal_end_effector_rot = hdf5_object['data'][f'demo_{trajectory_idx}']['next_obs']['robot0_eef_quat'][goal_state_idx]
    goal_gripper_joint_pos = hdf5_object['data'][f'demo_{trajectory_idx}']['next_obs']['robot0_gripper_qpos'][goal_state_idx]
    goal_object = hdf5_object['data'][f'demo_{trajectory_idx}']['next_obs']['object'][goal_state_idx]
    goal_state = np.concatenate([goal_end_effector_pos, goal_end_effector_rot, goal_gripper_joint_pos, goal_object], axis=-1)
    print(f"    - Goal State: {goal_state.shape}")
    return states, goal_state, actions

def plot_end_effector_trajectory(end_eff_pos, end_eff_goal_pos):
    """
    Plots the end effector trajectory for a given trajectory.

    Args:
        end_eff_pos (np.ndarray): array of end_eff_pos
        end_eff_goal_pos (np.ndarray): goal end_eff_goal_pos
        
    """

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plotting all states in blue
    ax.scatter(end_eff_pos[:, 0], end_eff_pos[:, 1], end_eff_pos[:, 2], marker='o', s=10, color='blue', label='Intermediate Poses')
    # Plotting the first state in red
    ax.scatter(end_eff_pos[0, 0], end_eff_pos[0, 1], end_eff_pos[0, 2], marker='o', s=30, color='red', label='Start Pose')
    # Plotting the last state in green
    ax.scatter(end_eff_goal_pos[0], end_eff_goal_pos[1], end_eff_goal_pos[2], marker='o', s=30, color='green', label='Goal Pose')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Adding a legend
    ax.legend()
    plt.show()