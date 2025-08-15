import os
import numpy as np
from scipy import stats
import pandas as pd
from scipy import signal
np.seterr(divide='ignore', invalid='ignore')


import pandas as pd
import os,requests,zipfile

import pdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def get_dataset(url:str, data_directory: str, file_name: str, unzip: bool):


    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
        
    response = requests.get(url, stream=True)
    data_file = open(os.path.join(data_directory, file_name), 'wb')
    total_size = int(response.headers.get("Content-Length", 0))
    chunk_size = 1024
    #print(data_file ); exit()
    for chunk in tqdm(iterable=response.iter_content(chunk_size=chunk_size), total=total_size / chunk_size,
                        unit='KB', unit_scale=True, unit_divisor=chunk_size):
        data_file.write(chunk)
    data_file.close()
        
    if unzip:
        print(f'Unzipping [{file_name}] ...')
        with zipfile.ZipFile(os.path.join(data_directory, file_name), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(data_directory, file_name.split('.')[0]))
            os.remove(os.path.join(data_directory, file_name))

        a='human_activity_recognition_using_smartphones/'
        b='UCI HAR Dataset.zip'
        c= 'human_activity_recognition_using_smartphones/UCI HAR Dataset/'

        print(f'Unzipping [{b}] ...')
        with zipfile.ZipFile(os.path.join(data_directory,a,b), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(data_directory,a))

    print('\n---DATASET DOWNLOAD COMPLETE---')
    return c

def preprocess_sbhar(load_path):
    """
    Process the UCI HAR dataset and save data organized by subject.
    
    Args:
        load_path (str): Path to the UCI HAR dataset root directory

    """
    
    save_path = os.path.join(load_path, 'processed_data') 
    os.makedirs(save_path, exist_ok=True)
    output_file = os.path.join(save_path, 'sub0_features.npy')
    if os.path.exists(output_file):
        print(f"Processed data already exists at {output_file}")
        return
    
    if not os.path.exists(os.path.join(load_path, 'uci_har_dataset', 'train')):
        print(f"Dataset not found at {load_path}. Downloading...")
        c= get_dataset(
            url='https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip',
            data_directory=load_path,
            file_name='human_activity_recognition_using_smartphones.zip',
            unzip=True
        )
    
    # Signal names (the order is important for consistency)
    signal_names = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z"
    ]
    
    # Dictionary to hold data for each subject
    subject_data = {}
    
    # Process both train and test sets
    for dataset in ['train', 'test']:
        print(f"Processing {dataset} dataset...")
        
        # Load subject IDs
        subject_file = os.path.join(load_path,c, dataset, f'subject_{dataset}.txt')
        subjects = pd.read_csv(subject_file, header=None, sep=r'\s+').values.flatten()
        
        # Load activity labels
        label_file = os.path.join(load_path,c, dataset, f'y_{dataset}.txt')
        labels = pd.read_csv(label_file, header=None, sep=r'\s+').values.flatten()
        labels-=1 # Convert to zero-based indexing
        
        # Load all signals
        signals_data = []
        for signal in signal_names:
            signal_file = os.path.join(load_path,c, dataset, 'Inertial Signals', f'{signal}_{dataset}.txt')
            # Read data with optimal settings for speed
            signal_data = pd.read_csv(
                signal_file, 
                header=None, 
                sep=r'\s+',
                dtype=np.float32  # Use float32 to save memory
            ).values
            signals_data.append(signal_data)
        
        # Stack signals along a new dimension
        # From list of (n_samples, 128) to (n_samples, 128, 9)
        stacked_signals = np.stack(signals_data, axis=-1)
        
        # Add data to the appropriate subject
        for subject_id in np.unique(subjects):
            # Get mask for this subject
            mask = subjects == subject_id
            
            # Get data for this subject
            subject_signals = stacked_signals[mask]
            subject_labels = labels[mask]
            
            # Initialize if this is the first time seeing this subject
            if subject_id not in subject_data:
                subject_data[subject_id] = {
                    'features': [],
                    'labels': []
                }
            
            # Append data
            subject_data[subject_id]['features'].append(subject_signals)
            subject_data[subject_id]['labels'].append(subject_labels)
    
    # Save data for each subject
    print("Saving data by subject...")
    for subject_id in tqdm(sorted(subject_data.keys())):
        # Concatenate data if there are multiple segments
        features = np.vstack(subject_data[subject_id]['features'])
        labels = np.concatenate(subject_data[subject_id]['labels'])
        
        # Save with zero-based indexing (subject 1 becomes 0, etc.)
        idx = subject_id - 1
        np.save(os.path.join(save_path, f'sub{idx}_features.npy'), features)
        np.save(os.path.join(save_path, f'sub{idx}_labels.npy'), labels)
        
        # Print information for verification
        print(f"Subject {subject_id} (idx {idx}): {features.shape[0]} samples, feature shape: {features.shape}, label shape: {labels.shape}")
    
    print(f"Processing complete. Files saved to {save_path}")
    return

