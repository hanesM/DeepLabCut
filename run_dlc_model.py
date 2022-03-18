import sys
from deeplabcut.pose_estimation_tensorflow.training import train_network
from deeplabcut.generate_training_dataset.multiple_individuals_trainingsetmanipulation import create_multianimaltraining_dataset

# Change config path depending on project location
config_path = sys.argv[1]

# Some training parameters
shuffle = 1
max_it = 15
save_it = 5
disp_it = 1


# Create datasets
create_multianimaltraining_dataset(config_path,Shuffles=[shuffle])

# Train network
train_network(config_path,shuffle=shuffle,displayiters=disp_it,saveiters=save_it, maxiters=max_it, max_snapshots_to_keep=None, allow_growth=True)