# imports
import bc_gcsl_model
import torch
import robomimic.utils.obs_utils as ObsUtils
from robomimic.config import config_factory
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import helper
from torch.utils.data import DataLoader
import taskBuffer
import numpy as np
import sys
# add pretrained_weights to the path
sys.path.append('../pretrained_weights')
# add data to the path
sys.path.append('../data')

##################################################
# Loading the model from pretrained phase
##################################################

lift_input_dim = 39 # 19 (state) + 19 (goal) + 1 (task identifier)
can_input_dim = 47 # 23 (state) + 23 (goal) + 1 (task identifier)
square_input_dim = 47 # 23 (state) + 23 (goal) + 1 (task identifier)
list_of_input_sizes = [lift_input_dim, can_input_dim, square_input_dim]
action_dim = 7 # 7 actions for the robot arm actuators

# checking if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# data paths
lift_data_path = '/project/data/lift/low_dim_v141.hdf5'
can_data_path = '/project/data/can/low_dim_v141.hdf5'
square_data_path = '/project/data/square/low_dim_v141.hdf5'

# setting up the encoder
encoder = bc_gcsl_model.DynamicEncoder(list_of_input_sizes)
# setting up the model
model = bc_gcsl_model.CustomBCModel(encoder, action_dim).to(device)

# loading the model from pretrained phase
model.load_state_dict(torch.load('pretrained_weights/pretrainedModel.pt'))
print("Model from pretrained phase loaded successfully!")

##################################################
# Generating trajectory for the iterative phase
##################################################


## Parameters to change
TASK_REPEAT_ITER = 20 # repeat task for <number> iterations
NUM_ITERATIONS = 10  # number of iterations to run the iterative phase
BATCH_SIZE = 128 # batch size for training the model
NUM_EPOCHS = 10 # number of epochs to train the model for the update phase
LR = 1e-5
Debug = False

# Fix NUM_ITERATIONS to be a multiple of TASK_REPEAT_ITER
NUM_ITERATIONS = NUM_ITERATIONS * TASK_REPEAT_ITER
# task mapping
tasks = {'lift':0, 'can':1, 'square':2}
data_paths = {'lift':lift_data_path, 'can':can_data_path, 'square':square_data_path}

# optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.MSELoss()

## environment configuration
# Now you can initialize observation utilities with this config
config = config_factory(algo_name="bc")
ObsUtils.initialize_obs_utils_with_config(config)

env_dict = {}  # Initialize environment dictionary

for task_name, task_id in tasks.items():
    # Load environment metadata

    # env_{task_name}_meta = FileUtils.get_env_metadata_from_dataset(data_paths[task_name])
    env_meta = FileUtils.get_env_metadata_from_dataset(data_paths[task_name])

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_meta['env_name'],
        render=False,
        render_offscreen=False,
        use_image_obs=False,
    )
    
    env_dict[f'env_{task_name}'] = env
    
# initialize task buffer
task_buffer = taskBuffer.TaskBuffer()

# initialize dataloader dictionary
dataloader_dict = {}

# Generate trajectories loop
for iteration_i in range(NUM_ITERATIONS//TASK_REPEAT_ITER):
    # Generate trajectory for the current task
    print("\n==================== Generating trajectories =================================")
    for task_name, task_id in tasks.items():
        for i in range(TASK_REPEAT_ITER):
            # Generate trajectory
            print(f"Generating trajectory: {i} for the {task_name} task")
            recorded_trajectory = helper.generate_trajectory(model,
                                                             env_dict[f'env_{task_name}'],
                                                             task_id, data_paths[task_name], horizon=150, selected_goal_ind=None, device=device, verbose=False)
            if Debug:
                print(f"states: {recorded_trajectory['states'].shape}")
                print(f"actions: {recorded_trajectory['actions'].shape}")
                print(f"goal_state: {recorded_trajectory['goal_state'].shape}")
                
            # take the last state as the goal state and replace the goal state with the last state
            recorded_trajectory['goal_state'] = np.full(recorded_trajectory['goal_state'].shape, recorded_trajectory['states'][-1])
            recorded_trajectory['states'] = recorded_trajectory['states'][:-1]
            
            if Debug:
                # Plotting the trajectory generated
                states = recorded_trajectory['states']
                goal_state = recorded_trajectory['goal_state']
                end_eff_pos = states[:, :3]
                end_eff_goal_pos = goal_state[-1,:3]
                helper.plot_end_effector_trajectory(end_eff_pos, end_eff_goal_pos)
            
            task_buffer.add_data(task_id, recorded_trajectory['states'], recorded_trajectory['goal_state'], recorded_trajectory['actions'])
            
    # Get data from the task buffer for updating the model
    print("\n************* Update GC model *************")
    for task_name, task_id in tasks.items():
        dataset = task_buffer.get_data(task_id)
        # Create data loader
        dataloader_dict[task_name] = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
        print(f"dataloader_{task_name}: {len(dataloader_dict[task_name])} batches of size {BATCH_SIZE}")
        
    # Update the model
    avg_loss_history, lift_loss_history, can_loss_history, square_loss_history, _ = bc_gcsl_model.train(model, 
                                                                                                        optimizer, 
                                                                                                        criterion, 
                                                                                                        num_epochs=NUM_EPOCHS, 
                                                                                                        data_loaders=dataloader_dict, 
                                                                                                        device=device,
                                                                                                        verbose=False, # Print losses for each task 
                                                                                                        model_save_path=f'checkpoints/gcModel_iteration_{iteration_i}.pt')