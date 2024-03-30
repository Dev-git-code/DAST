# import torch
# from torch.utils.data import Dataset, DataLoader
# from scipy.io import loadmat
from DAST_Network import DAST
import torch.nn as nn

# class JetEngineDataset(Dataset):
#     def __init__(self, file_path_data, file_path_labels, dataName , labelsName ):
#         # Load data from MATLAB files
#         data_mat = loadmat(file_path_data)
#         labels_mat = loadmat(file_path_labels)
#         # Extract data and labels from loaded MATLAB structures
#         self.data = data_mat[dataName]  # Assuming the variable name in the MATLAB file is 'data'
#         self.labels = labels_mat[labelsName]  # Assuming the variable name in the MATLAB file is 'labels'

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         # Return both input data and labels
#         return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)


# # Paths to your training and validation MATLAB files
# train_data_file = 'F001_window_size_trainX_new.mat'
# train_labels_file = 'slide_window_processed\F001_window_size_trainY.mat'
# validation_data_file = 'F001_window_size_testX_new.mat'
# validation_labels_file = 'slide_window_processed\F001_window_size_testY.mat'

# # Create instances of your dataset class
# train_dataset = JetEngineDataset(train_data_file, train_labels_file, 'train1X_new', 'train1Y')
# validation_dataset = JetEngineDataset(validation_data_file, validation_labels_file, 'test1X_new', 'test1Y')

# # Define your data loaders
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)




import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat

class JetEngineDataset(Dataset):
    def __init__(self, file_path_data, file_path_labels, dataName , labelsName ):
        # Load data from MATLAB files
        data_mat = loadmat(file_path_data)
        labels_mat = loadmat(file_path_labels)
        # Extract data and labels from loaded MATLAB structures
        self.data = torch.tensor(data_mat[dataName], dtype=torch.float32)
        self.labels = torch.tensor(labels_mat[labelsName], dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return both input data and labels
        return self.data[idx], self.labels[idx]

# Paths to your training and validation MATLAB files
train_data_file = 'F001_window_size_trainX_new.mat'
train_labels_file = 'slide_window_processed\F001_window_size_trainY.mat'
validation_data_file = 'F001_window_size_testX_new.mat'
validation_labels_file = 'slide_window_processed\F001_window_size_testY.mat'

# Create instances of your dataset class
train_dataset = JetEngineDataset(train_data_file, train_labels_file, 'train1X_new', 'train1Y')
validation_dataset = JetEngineDataset(validation_data_file, validation_labels_file, 'test1X_new', 'test1Y')

# Define your data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

# Check if data is loaded correctly
print("Training data shape:", train_dataset.data.shape)
print("Training labels shape:", train_dataset.labels.shape)
print("Validation data shape:", validation_dataset.data.shape)
print("Validation labels shape:", validation_dataset.labels.shape)

