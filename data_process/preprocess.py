import os
import shutil
import numpy as np
from scipy import stats
from pandas import Series
from sliding_window import sliding_window
import pandas as pd
from scipy import interpolate
from scipy import signal
import scipy.io
# ignpre numpy wanring
np.seterr(divide='ignore', invalid='ignore')


def preprocess_SBHAR(dataset_path, window, stride):
    
    load_dataset_path = dataset_path + 'RawData/'
    channel_num=3

    label_data = np.array(pd.read_csv(load_dataset_path + 'labels.txt', delim_whitespace=True, header=None))
    act_record = {}
    for exp_num in range(1,62):
        ind = np.where(label_data[:,0] == exp_num)
        act_record[exp_num] = label_data[ind,2:][0]
    
    user_data = {}
    user_label = {}
    for usr_idx in range(30):
        user_data[usr_idx] = np.empty([0, 128, 3], dtype=np.float)
        user_label[usr_idx] = np.empty([0], dtype=np.int)

    file_list = os.listdir(load_dataset_path)
    for file in file_list:
        if 'acc' in file:
            exp_num = int(file.split('_')[1][3:])
            usr_idx = int(file.split('_')[2][4:-4]) - 1 # 0-29 users
            data = np.array(pd.read_csv(load_dataset_path + file, delim_whitespace=True, header=None))
            
            # filtering
            for i in range(channel_num):
                data[:,i]=signal.medfilt(data[:,i], 3)# median filter
            sos = signal.butter(N=3, Wn=2*20/50, btype='lowpass', output='sos')# 3rd order low-pass Butter-worth filter with a 20 Hz cutoff frequency
            for i in range(channel_num):
                data[:,i] = signal.sosfilt(sos, data[:,i])
            sos = signal.butter(N=4, Wn=2*0.3/50, btype='highpass', output='sos')# separating gravity, low-pass Butterworth filter with a 0.3 Hz corner frequency
            for i in range(channel_num):
                data[:,i] = signal.sosfilt(sos, data[:,i])

            # normalization
            lower_bound = np.array([-0.5323077036833478, -0.4800314262209822, -0.4063855491288771])
            upper_bound = np.array([0.7359294642946127, 0.35672401151151384, 0.3462854467071975])
            diff = upper_bound - lower_bound
            data = 2 * (data - lower_bound) / diff - 1

            # generate labels
            label = np.ones(len(data)) * -1
            for act_seg in act_record[exp_num]:
                label[int(act_seg[1]-1):int(act_seg[2])] = act_seg[0] - 1 # label 0-11

            # sliding window
            data    = sliding_window( data, (window, channel_num), (stride, 1) )
            label   = sliding_window( label, window, stride )
            label   = stats.mode( label, axis=1 )[0][:,0]# choose the most common value as the label of the window

            invalid_idx = np.nonzero( label < 0 )[0]# remove invalid time windows (label==-1)
            data        = np.delete( data, invalid_idx, axis=0 )
            label       = np.delete( label, invalid_idx, axis=0 )

            user_data[usr_idx] = np.concatenate((user_data[usr_idx], data), axis=0)
            user_label[usr_idx] = np.concatenate((user_label[usr_idx], label), axis=0)
            print( "exp{} finished".format( exp_num) )

    for usr_idx in range(30):
        np.save( dataset_path + 'processed_data/' + 'sub{}_features'.format( usr_idx ), user_data[usr_idx] )
        np.save( dataset_path + 'processed_data/' + 'sub{}_labels'.format( usr_idx ), user_label[usr_idx] )  

def preprocess_opportunity(dataset_path, window, stride):

    file_list = [   ['S1-Drill.dat',
                    'S1-ADL1.dat',
                    'S1-ADL2.dat',
                    'S1-ADL3.dat',
                    'S1-ADL4.dat',
                    'S1-ADL5.dat'] ,
                    ['S2-Drill.dat',
                    'S2-ADL1.dat',
                    'S2-ADL2.dat',
                    'S2-ADL3.dat',
                    'S2-ADL4.dat',
                    'S2-ADL5.dat'] ,
                    ['S3-Drill.dat',
                    'S3-ADL1.dat',
                    'S3-ADL2.dat',
                    'S3-ADL3.dat',
                    'S3-ADL4.dat',
                    'S3-ADL5.dat'] ,
                    ['S4-Drill.dat',
                    'S4-ADL1.dat',
                    'S4-ADL2.dat',
                    'S4-ADL3.dat',
                    'S4-ADL4.dat',
                    'S4-ADL5.dat'] ]



    if os.path.exists( dataset_path + 'processed_data/' ):
        shutil.rmtree( dataset_path + 'processed_data/' )
    os.mkdir( dataset_path + 'processed_data/' )
    channel_num = 77
    for usr_idx in range( 4 ):
        
        print( "process data... user{}".format( usr_idx ) )
        time_windows    = np.empty( [0, window, channel_num], dtype=np.float64 )
        act_labels      = np.empty( [0], dtype=np.int64 )

        for file_idx in range( 5 ):
            
            filename = file_list[ usr_idx ][ file_idx ]

            file    = dataset_path +'dataset/'+ filename
            signal  = np.loadtxt( file )
            #import pdb; pdb.set_trace()
            index = [ 38, 39, 40, 41, 42, 43, 44, 45, 46, 51, 52, 53, 54, 55, 56, 57, 58, 59, 64, 65, 66, 67, 68, 69, 70, 71, 72, 77, 78, 79, 80, 81, 82, 83, 84, 85, 90, 91, 92, 93, 94, 95, 96, 97, 98, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134]
            index = [x - 1 for x in index]
            #index = [i for i in range(63, 66)]
            data = signal.take(index, axis=1)
            label = signal[:, 243].astype( np.int64 )

            label[ label == 0 ] = 0 # Null
            label[ label == 1 ] = 1 # Stand
            label[ label == 2 ] = 2 # Walk
            label[ label == 4 ] = 3 # Sit
            label[ label == 5 ] = 4 # Lie

            # fill missing values using Linear Interpolation
            data    = np.array( [Series(i).interpolate(method='linear') for i in data.T] ).T
            data[ np.isnan( data ) ] = 0.

            # upper_bound = np.array([975.0, 1348.0, 1203.0])
            # lower_bound = np.array([-1484.0, -493.0, -803.0])
            lowerBound = np.min(data, axis=(0, 1))  # Min across all samples and time steps
            upperBound = np.max(data, axis=(0, 1))  # Max across all samples and time steps
                # normalization
            diff = upperBound - lowerBound
            data = 2 * (data - lowerBound) / diff - 1

            data[ data > 1 ] = 1.0
            data[ data < -1 ] = -1.0

            #sliding window
            data    = sliding_window( data, (window, channel_num), (stride, 1) )
            label   = sliding_window( label, window, stride )
            
            label = stats.mode( label, axis=1 )[0] #[:,0]

            time_windows    = np.concatenate( (time_windows, data), axis=0 )
            act_labels      = np.concatenate( (act_labels, label), axis=0 )

        np.save( dataset_path + 'processed_data/' + 'sub{}_features'.format( usr_idx ), time_windows )
        np.save( dataset_path + 'processed_data/' + 'sub{}_labels'.format( usr_idx ), act_labels )                
        print( "sub{} finished".format( usr_idx) )  

def preprocess_pamap2(dataset_path, window, stride):

    axes = ['x', 'y', 'z']
    IMUsensor_columns = ['temperature'] + \
        ['acc_16g_' + i for i in axes] + \
        ['acc_6g_' + i for i in axes] + \
        ['gyroscope_' + i for i in axes] + \
        ['magnometer_' + i for i in axes] + \
        ['orientation_' + str(i) for i in range(4)]
    header = ["timestamp", "activityID", "heartrate"] + ["hand_" + s
                                                         for s in IMUsensor_columns] \
        + ["chest_" + s for s in IMUsensor_columns] + ["ankle_" + s
                                                       for s in IMUsensor_columns]

    if os.path.exists( dataset_path + 'processed_data/' ):
        shutil.rmtree( dataset_path + 'processed_data/' )
    os.mkdir( dataset_path + 'processed_data/' )

    file_list = [   'subject101.dat',
                    'subject102.dat',
                    'subject103.dat',
                    'subject104.dat',
                    'subject107.dat',
                    'subject108.dat',
                    'subject109.dat']

    columns_to_use = ['hand_acc_16g_x', 'hand_acc_16g_y', 'hand_acc_16g_z',
                    'ankle_acc_16g_x', 'ankle_acc_16g_y', 'ankle_acc_16g_z',
                    'chest_acc_16g_x', 'chest_acc_16g_y', 'chest_acc_16g_z']
    feature_columns= [ 1, 4, 5, 6, 10, 11, 12, 21, 22, 23, 27, 28, 29, 38, 39, 40, 44, 45, 46 ] #from https://github.com/saif-mahmud/self-attention-HAR/blob/main/configs/data.yaml
    channel_num = len(columns_to_use)- 1 # -1 because we don't use the label column

    for user in file_list:
        # load data
        datafram= pd.read_csv(dataset_path + user, sep=' ', header=None)
        datafram = datafram.iloc[:, feature_columns].fillna(0).interpolate() 
        datafram.columns = header()

        datafram = datafram[datafram.activityID != 0]
        unique_activity_ids = datafram['activityID'].unique()
        activity_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(unique_activity_ids))}
        datafram['activityID'] = datafram['activityID'].map(activity_id_map)

        features = datafram.drop(['activityID'], axis=1)
        labels = datafram['activityID']    
        data =features.values
        label = labels.values

        lowerBound = np.min(data, axis=(0, 1))  # Min across all samples and time steps
        upperBound = np.max(data, axis=(0, 1))  # Max across all samples and time steps
            # normalization
        diff = upperBound - lowerBound
        data = 2 * (data - lowerBound) / diff - 1

        data[ data > 1 ] = 1.0
        data[ data < -1 ] = -1.0

        #sliding window
        data    = sliding_window( data, (window, channel_num), (stride, 1) )
        label   = sliding_window( label, window, stride )

        label = stats.mode( label, axis=1 )[0] #[:,0]

        np.save( dataset_path + 'processed_data/' + 'sub{}_features'.format( user ), data )
        np.save( dataset_path + 'processed_data/' + 'sub{}_labels'.format( user ), label )
        print( "sub{} finished".format( user) )

    print( "~~~~~~~~~~All finished~~~~~~~" )

def preprocess_realworld(dataset_path, window, overlap):
    
    if os.path.exists( dataset_path + 'processed_data/' ):
        shutil.rmtree( dataset_path + 'processed_data/' )
    os.mkdir( dataset_path + 'processed_data/' )



    window_ms = window * 1000
    label_list = ['climbingdown', 'jumping', 'lying', 'running', 'sitting', 'standing', 'walking', 'climbingup']
    pos_list = ['chest']
    mod_list = ['acc']
    sen_list = []

    for usr_idx in range(15):
        data = []
        label = []
        path = os.path.join(dataset_path, 'proband' + str(usr_idx+1))
        for i in range(len(label_list)):
            act = label_list[i]
            sen_readers = []
            for pos in pos_list:
                for mod in mod_list:
                    if i == 0 and usr_idx == 0:
                        sen_list.append(pos + '_' + mod)
                    sen_readers.append(pd.read_csv(os.path.join(path, mod + '_' + act + '_' + pos + '.csv')))
            t_min = max([sen_readers[i]['attr_time'].min() for i in range(len(sen_readers))])
            t_max = min([sen_readers[i]['attr_time'].max() for i in range(len(sen_readers))])
            window_range = int((t_max - t_min - window_ms) / (window_ms * (1 - overlap)))
            min_size = window * 50 - 5          
            for idx in range(window_range):
                tmpx = []
                start_idx = t_min + idx * window_ms * (1 - overlap)
                stop_idx = start_idx + window_ms
                windows_list = [ sen_readers[i][(sen_readers[i]['attr_time'] >= (start_idx-30)) & (sen_readers[i]['attr_time'] <= (stop_idx+10))] for i in range(len(sen_readers)) ]
                windows_size = [ windows_list[i]['id'].size for i in range(len(sen_readers)) ]
                windows_mint = [ windows_list[i]['attr_time'].min() for i in range(len(sen_readers)) ]
                windows_maxt = [ windows_list[i]['attr_time'].max() for i in range(len(sen_readers)) ]
                if (min(windows_size) >= min_size) and (max(windows_mint) <= start_idx) and (min(windows_maxt) >= stop_idx-20):
                    # resample to even stepsize at 50Hz
                    new_time_steps = start_idx + 20 * np.arange(window * 50)
                    tmpx = np.hstack( [ resample(windows_list[i].iloc[:, 1].values, windows_list[i].iloc[:, 2:].values, new_time_steps) for i in range(len(sen_readers)) ] )
                    if tmpx is not None:
                        data.append(tmpx)
                        label.extend([i])                    
                else:
                    continue

        # normalization

        lowerBound = np.array( [-13.93775515,  -7.077859,     -14.2442133] )
        upperBound = np.array( [11.016919,     19.608511,     9.479243] )
        diff = upperBound - lowerBound
        data = 2 * (data - lowerBound) / diff - 1
        


        # Now normalize using these bounds
        data = np.array(data)  # Ensure data is numpy array

        data[ data > 1 ]    = 1.0
        data[ data < -1 ]   = -1.0

        np.save( dataset_path + 'processed_data/' + 'sub{}_features'.format( usr_idx ), data )
        np.save( dataset_path + 'processed_data/' + 'sub{}_labels'.format( usr_idx ), label )
        print( "sub{} finished".format( usr_idx) )  

def resample(x, y, xnew):
    f = interpolate.interp1d(x, y, kind='linear', axis=0)
    ynew = f(xnew)
    return ynew




def preprocess_hhar(dataset_path, window, stride):
    """
    Preprocess the Phones_accelerometer.csv dataset for deep learning.
    
    Parameters:
        dataset_path - path to the directory containing 'Phones_accelerometer.csv'
        window - size of the sliding window
        stride - step size for sliding the window
    
    Creates processed data files in dataset_path/processed_data/ directory
    """
    # Create processed_data directory if it doesn't exist
    processed_dir = os.path.join(dataset_path, 'processed_data')
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)
    os.makedirs(processed_dir)
    
    # Read the CSV file
    csv_path = os.path.join(dataset_path, 'Phones_accelerometer.csv')
    print(f"Reading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Drop rows with NaN ground truth
    df = df.dropna(subset=['gt'])
    
    # Get unique users and activities
    unique_users = df['User'].unique()
    unique_activities = df['gt'].unique()
    
    # Map activities to numeric labels (0 to N-1)
    activity_to_label = {activity: idx for idx, activity in enumerate(unique_activities)}
    # save activity_to_label to processed_dir directory
    np.save(os.path.join(processed_dir, 'activity_to_label.npy'), activity_to_label)
    
    # Extract only the needed columns
    feature_columns = ['x', 'y', 'z']
    channel_num = len(feature_columns)
    
    print(f"Processing data for {len(unique_users)} users with {len(unique_activities)} activities...")
    print(f"Activity mapping: {activity_to_label}")
    
    # Process data for each user
    for user_idx, user in enumerate(unique_users):
        print(f"Processing data for user {user} ({user_idx+1}/{len(unique_users)})")
        
        # Filter data for current user
        user_data = df[df['User'] == user].copy()
        
        if len(user_data) == 0:
            print(f"  No data for user {user}, skipping...")
            continue
        
        # Extract features and labels
        data = user_data[feature_columns].values
        labels = user_data['gt'].map(activity_to_label).values
        
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

