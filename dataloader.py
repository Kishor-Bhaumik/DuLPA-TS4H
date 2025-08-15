
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import argparse

class CombinedLoader:
    """Custom implementation of CombinedLoader with max_size_cycle mode"""
    
    def __init__(self, loaders_dict, mode="max_size_cycle"):
        self.loaders_dict = loaders_dict
        self.mode = mode
        # Calculate max_length during initialization
        self.max_length = max(len(loader) for loader in loaders_dict.values())
        
    def __iter__(self):
        # Create iterators for each loader
        self.iterators = {name: iter(loader) for name, loader in self.loaders_dict.items()}
        self.idx = 0
        return self
    
    def __next__(self):
        if self.idx >= self.max_length:
            raise StopIteration
            
        batch = {}
        # Get next batch from each loader, restart if necessary
        for name, iterator in self.iterators.items():
            try:
                batch[name] = next(iterator)
            except StopIteration:
                # Reset this iterator and get the first batch
                self.iterators[name] = iter(self.loaders_dict[name])
                batch[name] = next(self.iterators[name])
        
        self.idx += 1
        return batch
    
    def __len__(self):
        return self.max_length



def set_deterministic(seed):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    return torch.Generator().manual_seed(seed)



class BalancedAugmentedDataLoader(DataLoader):
    """DataLoader that balances samples across users and applies augmentation"""
    def __init__(self, dataset, **kwargs):
        self._dataset = dataset
        self.batch_size = kwargs.get('batch_size', 1)
        super().__init__(dataset, **kwargs)
        
    def __iter__(self):
        # Group indices by user
        user_indices = {}
        for idx in range(len(self._dataset)):
            _, _, user = self._dataset[idx]
            if user.item() not in user_indices:
                user_indices[user.item()] = []
            user_indices[user.item()].append(idx)
            
        # Create balanced batches
        while True:
            batch_indices = []
            for user in user_indices:
                # Get random indices for this user
                user_batch_idx = np.random.choice(
                    user_indices[user], 
                    size=self.batch_size, 
                    replace=False
                )
                batch_indices.extend(user_batch_idx)
            
            # Get the batch data
            x_batch, y_batch, u_batch = [], [], []
            for idx in batch_indices:
                x, y, u = self.dataset[idx]
                x_batch.append(x)
                y_batch.append(y)
                u_batch.append(u)
            
            # Stack and augment if needed
            x_batch = torch.stack(x_batch)
            yield (x_batch, 
                   torch.stack(y_batch), 
                   torch.stack(u_batch))




def get_dataset(batch_size, test_user, args=None):
    """
    Load and preprocess OPPORTUNITY dataset with optional augmentation
    
    Args:
        batch_size (int): Batch size for data loaders
        test_user (int): User ID for test data
        args: Configuration arguments
        
    Returns:
        train_loader_S: DataLoader for source domain (other users)
        train_loader_T: DataLoader for target domain (test user's training data)
        test_loader: DataLoader for test data (test user's test data)
    """
    generator = set_deterministic(args.seed)
    #assert args.dataset == 'OPPORTUNITY' or args.dataset == 'hhar' or , "Dataset not supported"
    
    path = args.dataset_path + 'processed_data'
    if args.use_group_testing:
        group_index = (test_user // len(args.subject_groups[0])) +1
        path = path + '/group_{}'.format(group_index)
        print("loading groups data from foldr : ", path)
        print("test user: ", test_user)


    adapt_ratio = 0.5
    window = args.window
    N_channels = args.num_channel
    N_users = args.N_users


    if isinstance(test_user, int):
        # Single test user
        test_users = [test_user]
    else:
        # List of test users
        test_users = test_user
    # Dictionary to store data for each source user
    source_users_data = {}

    x_T_combined = np.empty([0, window, N_channels], dtype=np.float64)
    y_T_combined = np.empty([0], dtype=np.int64)
    user_T_combined = np.empty([0], dtype=np.int64)

    
    
    for test_user_id in test_users:
        x_T = np.load(path+'/sub{}_features.npy'.format(test_user_id))
        y_T = np.load(path+'/sub{}_labels.npy'.format(test_user_id))
        user_T = np.full(len(y_T), test_user_id   )
        # Combine test data
        x_T_combined = np.concatenate((x_T_combined, x_T), axis=0)
        y_T_combined = np.concatenate((y_T_combined, y_T), axis=0)
        user_T_combined = np.concatenate((user_T_combined, user_T), axis=0)
        print(f"Test User {test_user_id}: {len(x_T)} samples")
    
    x_T, y_T, user_T = x_T_combined, y_T_combined, user_T_combined  

    min_samples = float('inf')
    #import pdb;pdb.set_trace()
    for user_idx in range(N_users):
        if user_idx not in test_users:
            curr_x = np.load(path+'/sub{}_features.npy'.format(user_idx))
            curr_y = np.load(path+'/sub{}_labels.npy'.format(user_idx))
            curr_user = np.full(len(curr_y), user_idx ) 
            source_users_data[user_idx] = {
                'x': curr_x,
                'y': curr_y,
                'user':  curr_user 
            }
            min_samples = min(min_samples, len(curr_y))

    # Calculate number of samples per user to maintain equal distribution
    samples_per_user = min_samples
    
    # Balance each source user's data
    x_S = np.empty([0, window, N_channels], dtype=np.float64)
    y_S = np.empty([0], dtype=np.int64)
    user_S = np.empty([0], dtype=np.int64)
    
    for user_idx, user_data in source_users_data.items():
        # Randomly sample to match the minimum size
        indices = np.random.choice(len(user_data['x']), samples_per_user, replace=False)
        #import pdb;pdb.set_trace()
        x_S = np.concatenate((x_S, user_data['x'][indices]), axis=0)
        y_S = np.concatenate((y_S, user_data['y'][indices]), axis=0)
        user_S = np.concatenate((user_S, user_data['user'][indices]), axis=0)
        print(f"Source User {user_idx}: {len(user_data['x'])} samples")

    # Split target data
    train_x_T, test_x_T, train_y_T, test_y_T, train_user_T, test_user_T = train_test_split(
        x_T, y_T, user_T, train_size=adapt_ratio, random_state=0)

    # Create source dataset
    train_dataset_S = TensorDataset(
        torch.from_numpy(x_S.astype(np.float32)), 
        torch.from_numpy(y_S),
        torch.from_numpy(user_S))
    
    # Create target dataset
    train_dataset_T = TensorDataset(
        torch.from_numpy(train_x_T.astype(np.float32)), 
        torch.from_numpy(train_y_T),
        torch.from_numpy(train_user_T))
        
    # Create test dataset
    test_dataset = TensorDataset(
        torch.from_numpy(test_x_T.astype(np.float32)),
        torch.from_numpy(test_y_T),
        torch.from_numpy(test_user_T))


    # Use custom balanced loader without augmentation
    train_loader_S = BalancedAugmentedDataLoader(
        train_dataset_S, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True, 
        num_workers=2
    )
    
    train_loader_T = DataLoader(
        train_dataset_T,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        generator=generator
    )

    # Standard test loader (no augmentation needed)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=True, 
        num_workers=2,
        generator=generator
    )

    return train_loader_S, train_loader_T, test_loader


    
