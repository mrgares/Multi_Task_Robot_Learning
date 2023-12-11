import os
import robomimic.utils.file_utils as FileUtils
from robomimic import DATASET_REGISTRY
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
from torch.utils.data import Dataset
import torch
from torch.nn.functional import mse_loss


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

def extract_trajectory_i(hdf5_object, trajectory_idx, verbose=True):
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
    
    if verbose:
        print(f"Extracting Trajectory {trajectory_idx}:")
    
    # Actions
    actions = np.array(hdf5_object['data'][f'demo_{trajectory_idx}']['actions'])
    if verbose:
        print(f"    - Actions: {actions.shape}")
    
    # states
    end_effector_pos = hdf5_object['data'][f'demo_{trajectory_idx}']['obs']['robot0_eef_pos']
    end_effector_rot = hdf5_object['data'][f'demo_{trajectory_idx}']['obs']['robot0_eef_quat']
    gripper_joint_pos = hdf5_object['data'][f'demo_{trajectory_idx}']['obs']['robot0_gripper_qpos']
    object = hdf5_object['data'][f'demo_{trajectory_idx}']['obs']['object']
    states = np.concatenate([end_effector_pos, end_effector_rot, gripper_joint_pos, object], axis=-1)
    if verbose:
        print(f"    - States: {states.shape}")
    
    # goal state
    goal_state_idx = states.shape[0] - 1
    goal_end_effector_pos = hdf5_object['data'][f'demo_{trajectory_idx}']['next_obs']['robot0_eef_pos'][goal_state_idx]
    goal_end_effector_rot = hdf5_object['data'][f'demo_{trajectory_idx}']['next_obs']['robot0_eef_quat'][goal_state_idx]
    goal_gripper_joint_pos = hdf5_object['data'][f'demo_{trajectory_idx}']['next_obs']['robot0_gripper_qpos'][goal_state_idx]
    goal_object = hdf5_object['data'][f'demo_{trajectory_idx}']['next_obs']['object'][goal_state_idx]
    goal_state = np.concatenate([goal_end_effector_pos, goal_end_effector_rot, goal_gripper_joint_pos, goal_object], axis=-1)
    if verbose:
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
    
def aggregate_data(hdf5_path, task_id, verbose=False):
    """ 
    Aggregates all the data from the HDF5 file into a single array.
    
    Args:
        hdf5_path (str): path to the HDF5 file
        task_id (int): task id to use for the dataset
            - 0: lift
            - 1: Can
    Returns:
        states (np.ndarray): array of states
        actions (np.ndarray): array of actions
    """
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        all_states, all_goals, all_actions = [], [], []
        num_trajectories = len(hdf5_file['data'])
        for i in range(num_trajectories):
            states, goal_state, actions = extract_trajectory_i(hdf5_file, i, verbose=verbose)
            goal_state_replicated = np.tile(goal_state, (states.shape[0], 1))
            task_ids = np.full((states.shape[0], 1), task_id)
            all_states.append(np.concatenate([states, goal_state_replicated, task_ids], axis=-1))
            all_actions.append(actions)
        return np.concatenate(all_states, axis=0), np.concatenate(all_actions, axis=0)

class CustomDataset(Dataset):
    def __init__(self, inputs, actions):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.actions[idx]

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs, actions = self.data[idx]
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.float32)
        return inputs_tensor, actions_tensor

def obs_to_input(obs, goal_state, task_id, device):
    end_effector_pos = obs['robot0_eef_pos']
    end_effector_rot = obs['robot0_eef_quat']
    gripper_joint_pos = obs['robot0_gripper_qpos']
    object = obs['object']
    # Concatenate the observation, goal state and task id
    input = np.concatenate([end_effector_pos, end_effector_rot, gripper_joint_pos, object, goal_state, [task_id]])
    # convert to tensor
    input_tensor = torch.tensor(np.array([input]), dtype=torch.float32).to(device)
    return input_tensor

def obs_to_state(obs):
    end_effector_pos = obs['robot0_eef_pos']
    end_effector_rot = obs['robot0_eef_quat']
    gripper_joint_pos = obs['robot0_gripper_qpos']
    object = obs['object']
    # Concatenate the observation
    state = np.concatenate([end_effector_pos, end_effector_rot, gripper_joint_pos, object])
    return state
    
def custom_run_rollout_and_evaluate(model, env, task_id, horizon, goal_state, device='cpu'):
    obs = env.reset()
    done = False
    step = 0
    total_reward = 0
    while not done and step < horizon:
        # Prepare observation for model
        obs_tensor = obs_to_input(obs, goal_state, task_id, device)
        action = model(obs_tensor, task_id).squeeze(0).cpu().detach().numpy()
        
        # Take action in environment
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step += 1

    # Final state
    final_state = np.concatenate([obs[key] for key in ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'object']])

    # Calculate MSE loss
    mse = mse_loss(torch.tensor(final_state), torch.tensor(goal_state)).item()
    return mse, total_reward

def evaluate_model(model, env, task_id, data_path, horizon, num_rollouts=10, device='cpu', verbose=False):
    total_mse = []
    cumulative_reward = []
    for traj_idx in range(num_rollouts):
        with h5py.File(data_path, 'r') as hdf5_file:
            _, goal_state, _ = extract_trajectory_i(hdf5_file, trajectory_idx=traj_idx, verbose=verbose)

        mse_i, reward_i = custom_run_rollout_and_evaluate(model, env, task_id=task_id, horizon=horizon, goal_state=goal_state, device=device)
        total_mse.append(mse_i)
        cumulative_reward.append(reward_i)
        print(f"Trajectory {traj_idx}, MSE: {mse_i}, Reward: {reward_i}")
    average_mse = np.mean(total_mse)
    average_reward = np.mean(cumulative_reward)
    print(f"Average MSE over {num_rollouts} rollouts: {average_mse}, Average Reward: {average_reward}")
    return total_mse, cumulative_reward

def generate_trajectory(model, env, task_id, data_path, horizon=200, selected_goal_ind=None, device='cpu', verbose=False):
    with h5py.File(data_path, 'r') as hdf5_file:
        if selected_goal_ind is None:
            idx = np.random.randint(0, len(hdf5_file["data"]))
        else:
            idx = selected_goal_ind
        _, goal_state, _ = extract_trajectory_i(hdf5_file, trajectory_idx=idx, verbose=verbose)
    obs = env.reset()
    done = False
    step = 0
    record_trajectory = {'states': [obs_to_state(obs)], 'actions': [], 'goal_state': []}
    while not done and step < horizon:
        # Prepare observation for model
        
        record_trajectory['goal_state'].append(goal_state)
        obs_tensor = obs_to_input(obs, goal_state, task_id, device)
        action = model(obs_tensor, task_id).squeeze(0).cpu().detach().numpy()
        
        # Take action in environment
        obs, reward, done, info = env.step(action)
        record_trajectory['states'].append(obs_to_state(obs))
        record_trajectory['actions'].append(action)
        step += 1
    # convert to numpy array
    for key in record_trajectory.keys():
        record_trajectory[key] = np.array(record_trajectory[key])
    return record_trajectory