
import numpy as np
import os
import glob

def create_dictionary(main_list, v):
    """
    Create a dictionary mapping subjects to numeric values
    Args:
        main_list: List of subject group lists
        v: Current test group
    Returns:
        Dictionary mapping subjects to values
    """
    result_dict = {}
    value_counter = 0
    for sublist in main_list:
        if sublist == v:
            result_dict[tuple(v)] = value_counter  # Use tuple as list is not hashable
            value_counter += 1
        else:
            for item in sublist:
                result_dict[str(item)] = value_counter
                value_counter += 1
    return result_dict

def organize_files(path, subject_groups):
    """
    Process USC-HAD dataset for cross-validation
    
    Args:
        path (str): Path to the processed data directory
        subject_groups (list): List of lists containing subject IDs for cross-validation groups
        
    Returns:
        None: Files are saved to disk in group folders
    """
    path = os.path.join(path, 'processed_data/')
    # Get all files in the directory
    files = glob.glob(path + '*')

    # Process each group for cross-validation
    for group_idx, test_group in enumerate(subject_groups):
        print(f"Group {group_idx + 1}: {test_group}")
        
        # Create group folder
        group_folder = os.path.join(path, f'group_{group_idx + 1}')
        if not os.path.exists(group_folder):
            os.makedirs(group_folder)

        # Step 1: Process test data (concatenate data from test subjects)
        test_x, test_y = [], []
        test_files = []

        for test_sub in test_group:
            # Load test subject data
            tx = np.load(os.path.join(path, f'sub{test_sub}_features.npy'))
            ty = np.load(os.path.join(path, f'sub{test_sub}_labels.npy'))
            
            # Add to test data collections
            test_x.append(tx)
            test_y.append(ty)
            
            # Keep track of test files
            test_files.append(os.path.join(path, f'sub{test_sub}_features.npy'))
            test_files.append(os.path.join(path, f'sub{test_sub}_labels.npy'))

        # Concatenate all test data
        test_x = np.concatenate(test_x, axis=0)
        test_y = np.concatenate(test_y, axis=0)

        # Create mapping for file naming
        filename_dict = create_dictionary(subject_groups, test_group)

        # Step 2: Save the concatenated test data
        np.save(os.path.join(group_folder, f'sub{filename_dict[tuple(test_group)]}_features.npy'), test_x)
        np.save(os.path.join(group_folder, f'sub{filename_dict[tuple(test_group)]}_labels.npy'), test_y)
        
        # Step 3: Process and save non-test files 
        new_files = [f for f in files if f not in test_files and 'sub' in f]
        
        # Save each non-test file with new naming scheme
        for file in new_files:
            for key, value in filename_dict.items():
                if f'sub{key}_features.npy' in file:
                    new_file_name = os.path.join(group_folder, f'sub{value}_features.npy')
                    np.save(new_file_name, np.load(file))

                if f'sub{key}_labels.npy' in file:
                    new_file_name = os.path.join(group_folder, f'sub{value}_labels.npy')
                    np.save(new_file_name, np.load(file))
