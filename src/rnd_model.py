import torch
import torch.nn as nn
import numpy as np
import helper
import h5py
from torch.utils.data import Dataset
import itertools

class DynamicEncoder(nn.Module):
    def __init__(self, list_of_input_sizes):
        super(DynamicEncoder, self).__init__()
        for i, input_size in enumerate(list_of_input_sizes):
            setattr(self, f"task_{i}", nn.Linear(input_size, 128))
        self.common_fc = nn.Linear(128, 8)

    def forward(self, x, task_id):
        x = getattr(self, f"task_{task_id}")(x)
        return torch.relu(self.common_fc(x))

class RNDNetwork(nn.Module):
    def __init__(self, encoder, output_dim):
        super(RNDNetwork, self).__init__()
        self.encoder = encoder
        self.fc1 = nn.Linear(8, 8)
        # self.fc2 = nn.Linear(1024, 1024)
        self.output = nn.Linear(8, output_dim)

    def forward(self, x, task_id):
        x = self.encoder(x, task_id)
        x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        return self.output(x)

class RNDModel(nn.Module):
    def __init__(self, list_of_input_sizes, output_dim):
        super(RNDModel, self).__init__()
        
        # Create separate instances of DynamicEncoder for target and prediction networks
        target_encoder = DynamicEncoder(list_of_input_sizes)
        prediction_encoder = DynamicEncoder(list_of_input_sizes)
        
        self.target_network = RNDNetwork(target_encoder, output_dim)
        self.prediction_network = RNDNetwork(prediction_encoder, output_dim)

        # Freeze the target network to prevent training
        for param in self.target_network.parameters():
            param.requires_grad = False

    def forward(self, x, task_id):
        target_output = self.target_network(x, task_id)
        prediction_output = self.prediction_network(x, task_id)
        return target_output, prediction_output
    
##################################################
# Helper functions
##################################################

def train(model, optimizer, criterion, num_epochs, data_loaders, device, model_save_path, verbose=False):
    """
    Train the model for one epoch.
    """
    
    dataloader_indices = list(range(len(data_loaders)))
    map_task_id_to_task_name = {0: "lift", 1: "can", 2: "square"}
    dataloaders_cycle = itertools.cycle(dataloader_indices)
    
    # Determine the maximum number of batches among the DataLoaders
    max_batches = max([len(data_loader) for data_loader in data_loaders.values()])

    # Initialize a list to store the last loss for each task
    last_losses = [None] * len(data_loaders)
    avg_loss_history = []
    lift_loss_history = []
    can_loss_history = []
    square_loss_history = []
    
    model.train()
    for epoch in range(num_epochs):
        # Reset the iterator for each DataLoader at the start of each epoch
        dataloader_iters = [iter(loader) for loader in data_loaders.values()]
        
        for batch_idx in range(max_batches * len(data_loaders)):
            # Get the next DataLoader index in the cycle
            current_dataloader_idx = next(dataloaders_cycle)
            # Get the next batch from the DataLoader
            current_dataloader_iter = dataloader_iters[current_dataloader_idx]

            try:
                inputs = next(current_dataloader_iter)
            except StopIteration:
                # This DataLoader is exhausted, skip to the next one
                continue
            
            optimizer.zero_grad()
            
            # Forward pass
            target, prediction = model(inputs.to(device), current_dataloader_idx)
            # Backward pass
            loss = criterion(target, prediction)
            loss.backward()
            optimizer.step()
            
            # Update the last loss for the current task
            last_losses[current_dataloader_idx] = loss.item()
            # print initial losses for epoch 0
            if epoch == 0 and batch_idx == len(data_loaders) - 1:
                print(f"Initial losses: {last_losses}, avg loss: {np.mean(last_losses)}")
        avg_loss_history.append(np.mean(last_losses))
        lift_loss_history.append(last_losses[0])
        can_loss_history.append(last_losses[1])
        square_loss_history.append(last_losses[2])
        
        print(f"Epoch {epoch+1}, Avg Loss: {np.mean(last_losses):.4f}")
        if verbose:
            for i, task_loss in enumerate(last_losses):
                print(f"Task {i+1} Last Loss: {task_loss:.4f}")
            
    # Save model
    if model_save_path is not None:
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved at {model_save_path}")
    return avg_loss_history, lift_loss_history, can_loss_history, square_loss_history, model

    
def aggregate_data(hdf5_path, task_id, verbose=False, get_goal_states=False):
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
        all_states, all_goals = [], []
        num_trajectories = len(hdf5_file['data'])
        for i in range(num_trajectories):
            states, goal_state, actions = helper.extract_trajectory_i(hdf5_file, i, verbose=verbose)
            task_ids = np.full((states.shape[0], 1), task_id)
            all_states.append(np.concatenate([states, task_ids], axis=-1))
            all_goals.append(goal_state)
        if get_goal_states == True:
            all_goals = np.reshape(np.array(all_goals), (-1, all_goals[0].shape[0]))
            goal_task_id = np.full((all_goals.shape[0], 1), task_id)
            return np.concatenate(all_states, axis=0), np.concatenate([all_goals, goal_task_id], axis=-1)
        return np.concatenate(all_states, axis=0)

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs= self.data[idx]
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
        return inputs_tensor
    