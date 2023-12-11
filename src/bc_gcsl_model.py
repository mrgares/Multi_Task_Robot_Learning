import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import itertools


class DynamicEncoder(nn.Module):
    def __init__(self, list_of_input_sizes):
        super(DynamicEncoder, self).__init__()
        for i, input_size in enumerate(list_of_input_sizes):
            setattr(self, f"task_{i}", nn.Linear(input_size, 128))
        self.common_fc = nn.Linear(128, 128)

    def forward(self, x, task_id):
        x = getattr(self, f"task_{task_id}")(x)
        return torch.relu(self.common_fc(x))

class CustomBCModel(nn.Module):
    def __init__(self, encoder, action_dim):
        super(CustomBCModel, self).__init__()
        self.encoder = encoder
        self.fc1 = nn.Linear(128, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.output = nn.Linear(1024, action_dim)

    def forward(self, x, task_id):
        x = self.encoder(x, task_id)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.output(x)
    
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
                inputs, actions = next(current_dataloader_iter)
            except StopIteration:
                # This DataLoader is exhausted, skip to the next one
                continue
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(inputs.to(device), current_dataloader_idx)
            # Backward pass
            loss = criterion(predictions, actions.to(device))
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
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")
    return avg_loss_history, lift_loss_history, can_loss_history, square_loss_history, model