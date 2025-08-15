import os,requests,zipfile,io
import numpy as np
from scipy import stats
from pandas import Series
from sliding_window import sliding_window
import pandas as pd
import scipy
from tqdm import tqdm
import tarfile


def read_dir(directory):
    subject = []
    act_num = []
    sensor_readings = []
    for path, subdirs, files in os.walk(directory):
        for name in files:
            if name.endswith('.mat'):
                mat = scipy.io.loadmat(os.path.join(path, name))
                subject.extend(mat['subject'])
                sensor_readings.append(mat['sensor_readings'])

                if mat.get('activity_number') is None:
                    act_num.append('11')
                else:
                    act_num.append(mat['activity_number'])
    return subject, act_num, sensor_readings



def read_wisdm(DATA_PATH, save_csv= True):

    COLUMN_NAMES = [
        'subject',
        'activity',
        'x-axis',
        'y-axis',
        'z-axis'
    ]

    # Process the file line by line
    processed_lines = []
    text_file= os.path.join(DATA_PATH, 'WISDM_ar_latest/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt')
    with open(text_file, 'r') as file:
        content = file.read()
        # Split by semicolons to get individual records
        records = content.split(';')
        
        for record in records:
            record = record.strip()
            if not record:  # Skip empty records
                continue
                
            # Split by commas to get fields
            fields = record.split(',')
            if len(fields) == 6:  # Only keep records with exactly 6 fields
                # Skip the timestamp field (index 2)
                filtered_fields = [fields[0], fields[1], fields[3], fields[4], fields[5]]
                processed_lines.append(','.join(filtered_fields))

    # Create a string with valid CSV data
    valid_csv = '\n'.join(processed_lines)
    data_io = io.StringIO(valid_csv)

    # Read with pandas - no need to specify timestamp in column names
    data = pd.read_csv(data_io, header=None, names=COLUMN_NAMES)

    if save_csv:
        saving_path = DATA_PATH+'/processed_data/wisdm.csv' 
        if not os.path.exists(os.path.dirname(saving_path)):
            print('Creating directory for saving csv')
            os.makedirs(os.path.dirname(saving_path))
        print('Saving data to csv.....')
        data.to_csv(saving_path, index=False)
        print('Data saved to csv')


def get_dataset(url:str, data_directory: str, file_name: str, unzip: bool):


    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
        
    response = requests.get(url, stream=True)
    data_file = open(os.path.join(data_directory, file_name), 'wb')
    total_size = int(response.headers.get("Content-Length", 0))
    chunk_size = 1024

    for chunk in tqdm(iterable=response.iter_content(chunk_size=chunk_size), total=total_size / chunk_size,
                        unit='KB', unit_scale=True, unit_divisor=chunk_size):
        data_file.write(chunk)
    data_file.close()
        
    if unzip:
        print(f'Unzipping [{file_name}] ...')

        with tarfile.open(os.path.join(data_directory, file_name), 'r:gz') as tar:
            tar.extractall(os.path.join(data_directory, file_name.split('.')[0]))
            os.remove(os.path.join(data_directory, file_name))

    print('\n---DATASET DOWNLOAD COMPLETE---')
    return 


def preprocess_wisdm(dataset_path, window, stride):
    
    processed_dir = dataset_path + 'processed_data/'
    if not os.path.exists( os.path.join(processed_dir, 'usc-had.csv')):
        get_dataset(
            url='https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz',
            data_directory=dataset_path,
            file_name='WISDM_ar_latest.tar.gz',
            unzip=True)

        read_wisdm(os.path.join(dataset_path), save_csv=True)

    csv_path = os.path.join(processed_dir, 'wisdm.csv')
    
    print(f"Reading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Drop rows with NaN ground truth
    df = df.dropna(subset=['activity'])
    
    # Get unique users and activities
    unique_users = df['subject'].unique()
    unique_activities = df['activity'].unique()
    
    # Map activities to numeric labels (0 to N-1)
    activity_to_label = {activity: idx for idx, activity in enumerate(unique_activities)}
    np.save(os.path.join(processed_dir, 'activity_to_label.npy'), activity_to_label)
    
    # Extract only the needed columns
    feature_columns = [ 'x-axis','y-axis','z-axis']
    channel_num = len(feature_columns)
    
    print(f"Processing data for {len(unique_users)} users with {len(unique_activities)} activities...")
    print(f"Activity mapping: {activity_to_label}")
    
    # Process data for each user
    for user_idx, user in enumerate(unique_users):
        print(f"Processing data for user {user} ({user_idx+1}/{len(unique_users)})")
        
        # Filter data for current user
        user_data = df[df['subject'] == user].copy()
        
        if len(user_data) == 0:
            print(f"  No data for user {user}, skipping...")
            continue
        
        # Extract features and labels
        data = user_data[feature_columns].values
        labels = user_data['activity'].map(activity_to_label).values
        
        # Check if we have enough data for the given window size
        if len(data) < window:
            print(f"  Not enough data for user {user} to create a window of size {window}, skipping...")
            continue
        
        # Fill missing values using linear interpolation
        data = np.array([Series(i).interpolate(method='linear') for i in data.T]).T
        data[np.isnan(data)] = 0.  # Replace any remaining NaNs with 0
        
        # Normalize the data to [-1, 1] range
        lower_bound = np.min(data, axis=0)
        upper_bound = np.max(data, axis=0)
        diff = upper_bound - lower_bound
        
        # Avoid division by zero
        diff[diff == 0] = 1
        
        data = 2 * (data - lower_bound) / diff - 1
        
        # Clip values to ensure they're in the [-1, 1] range
        data[data > 1] = 1.0
        data[data < -1] = -1.0
        
        # Apply sliding window
        #use_slide_window = True
        try:
            windowed_data = sliding_window(data, (window, channel_num), (stride, 1))
            windowed_labels = sliding_window(labels, window, stride)
            
            # Get the most frequent label for each window
            window_labels = stats.mode(windowed_labels, axis=1, keepdims=False)[0]
            
            # Save the processed data
            np.save(os.path.join(processed_dir, f'sub{user_idx}_features'), windowed_data)
            np.save(os.path.join(processed_dir, f'sub{user_idx}_labels'), window_labels)
            
            print(f"  Saved {len(windowed_data)} windows for user {user}")
            
        except ValueError as e:
            print(f"  Error processing windows for user {user}: {e}")
            continue
    
    print("Preprocessing complete!")





