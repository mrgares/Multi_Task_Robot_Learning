import numpy as np
import helper
import rnd_model

class TaskBuffer:
    def __init__(self):
        # Initialize dictionaries to store states, goal_states, and actions for each task_id
        self.net_inputs = {}
        self.actions = {}

    def add_data(self, task_id, states, goal_states, actions):
        """
        Add states, goal_states, and actions for a specific task_id.
        """
        # If the task_id does not exist in the dictionaries, initialize empty lists for it
        if task_id not in self.net_inputs:
            self.net_inputs[task_id] = []
            self.actions[task_id] = []

        task_ids = np.full((states.shape[0], 1), task_id)
        # Append the new data to the respective lists
        self.net_inputs[task_id].append(np.concatenate([states, goal_states, task_ids], axis=1))
        self.actions[task_id].append(actions)

    def get_data(self, task_id, reset=False):
        """
        Retrieve states, goal_states, and actions for a specific task_id.
        """
        self.net_inputs[task_id] = np.concatenate(self.net_inputs[task_id], axis=0)
        self.actions[task_id] = np.concatenate(self.actions[task_id], axis=0)
        dataset = helper.CustomDataset(list(zip(self.net_inputs[task_id], self.actions[task_id])))
        # reset the buffer
        if reset:
            self.net_inputs[task_id] = []
            self.actions[task_id] = []
        return dataset
    
    def get_data_rnd(self, task_id, reset=False):
        """
        Retrieve states for a specific task_id.
        """
         
        if task_id == 0:
            states = self.net_inputs[task_id][:, :19]
            ids = (self.net_inputs[task_id][:, -1]).reshape(-1, 1)
            dataset = rnd_model.CustomDataset(np.concatenate([states, ids], axis=1))
            if reset:
                self.net_inputs[task_id] = []
                self.actions[task_id] = []
            # output should be of shape (num_samples, 20)
            return dataset
        elif task_id == 1:
            states = self.net_inputs[task_id][:, :23]
            ids = (self.net_inputs[task_id][:, -1]).reshape(-1, 1)
            dataset = rnd_model.CustomDataset(np.concatenate([states, ids], axis=1))
            if reset:
                self.net_inputs[task_id] = []
                self.actions[task_id] = []
            # output should be of shape (num_samples, 24)
            return dataset
        elif task_id == 2:
            states = self.net_inputs[task_id][:, :23]
            ids = (self.net_inputs[task_id][:, -1]).reshape(-1, 1)
            dataset = rnd_model.CustomDataset(np.concatenate([states, ids], axis=1))
            if reset:
                self.net_inputs[task_id] = []
                self.actions[task_id] = []
            # output should be of shape (num_samples, 24)
            return dataset