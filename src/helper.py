import os
import robomimic.utils.file_utils as FileUtils
from robomimic import DATASET_REGISTRY
import numpy as np


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
    Prints information about a trajectory in the HDF5 file.

    Args:
        hdf5_object: h5py.File or h5py.Group object
        trajectory_idx (int): index of the trajectory to print
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