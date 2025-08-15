import os
import shutil
import numpy as np
from scipy import stats
from pandas import Series
from sliding_window import sliding_window
import requests
from tqdm import tqdm
import zipfile
np.seterr(divide='ignore', invalid='ignore')


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
        
    print('\n---DATASET DOWNLOAD COMPLETE---')
    return
            
            
    

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

    FILE= 'OPPORTUNITY'
    #f not os.path.exists( dataset_path ):
    URL= 'https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip'
    if not os.path.exists(dataset_path):
        print("\nDOWNLOADING OPPORTUNITY DATASET...........\n")
        get_dataset(url=URL, data_directory= dataset_path, file_name=FILE+".zip" , unzip=True)
        
    
    save_path = os.path.join(dataset_path, 'processed_data/')
    os.makedirs(save_path, exist_ok=True)
    
    loading_path = os.path.join(dataset_path, FILE)
    # if os.path.exists( dataset_path + 'processed_data/' ):
    #     shutil.rmtree( dataset_path + 'processed_data/' )
    
    channel_num = 77
    print("Preprocessing OPPORTUNITY dataset...\n")
    for usr_idx in range( 4 ):
        
        print( "process data... user{}".format( usr_idx +1 ) )
        time_windows    = np.empty( [0, window, channel_num], dtype=np.float64 )
        act_labels      = np.empty( [0], dtype=np.int64 )

        for file_idx in range( 5 ):
            
            filename = file_list[ usr_idx ][ file_idx ]
            file  = os.path.join(loading_path, 'OpportunityUCIDataset/dataset/', filename )

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

        np.save( save_path + 'sub{}_features'.format( usr_idx ), time_windows )
        np.save( save_path  + 'sub{}_labels'.format( usr_idx ), act_labels )                
        print( "sub{} finished".format( usr_idx) )  