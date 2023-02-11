import librosa
import numpy as np
import pandas as pd

# Audio Config
DURATION = 4
SAMPLE_RATE = 16000
AUDIO_SHAPE = SAMPLE_RATE*DURATION

NOISE_DIM = 500
MFCC = 40

ENCODE_SIZE = NOISE_DIM
DENSE_SIZE = 2100

# Paths
DATASET_PATH      = "/Users/surendrasrinivas/Downloads/B.TECH/SEMESTER 6/temp/Audio_Generation/Datasets/"
#AUTO_ENCODER_PATH = "./WavFiles/Autoencoder/"
AUTO_ENCODER_PATH = "/Users/surendrasrinivas/Downloads/B.TECH/SEMESTER 6/temp/Audio_Generation/Datasets/WavFiles/Autoencoder/"
#PICTURE_PATH      = "./Pictures/"
PICTURE_PATH = "/Users/surendrasrinivas/Downloads/B.TECH/SEMESTER 6/temp/Audio_Generation/Datasets/Pictures/"
#GAN_PATH          = "./WavFiles/GAN/"
GAN = "/Users/surendrasrinivas/Downloads/B.TECH/SEMESTER 6/temp/Audio_Generation/GAN/"

#Label
LABEL = "Violin_or_fiddle"
#LABEL = "Laughter Hi-Hat Flute"
#LABEL = "gender"

# Load 
def load_train_data(input_length=AUDIO_SHAPE, label = LABEL):
#def load_train_data(input_length=AUDIO_SHAPE, gender = LABEL):

    train = pd.read_csv(DATASET_PATH + "train.csv")
    #train = pd.read_csv(DATASET_PATH+"cv-valid-train.csv")
    if label != None:
    #if gender != None:
        train_list = train.loc[train.label == label]
        #train_list = train.loc[train.gender == gender]
    else: 
        train_list = train
    cur_batch_size = len(train_list)
    train_fname_list = train_list.fname
    #train_fname_list = train.filename
    X = np.empty((cur_batch_size, input_length))
    for i, train_fname in enumerate(train_fname_list):
        file_path = DATASET_PATH + "audio_train/" + train_fname
        #file_path = DATASET_PATH + train_fname
        
        # Read and Resample the audio
        data, _ = librosa.core.load(file_path, sr=SAMPLE_RATE, res_type='kaiser_fast')

        # Random offset / Padding
        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length+offset)]
        else:
            if input_length > len(data):
                max_offset = input_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data, (offset, input_length - len(data) - offset), "constant")
        X[i,] = data
    print("Data Loaded...")
    return X

# Stardize Data 
def normalization(X):
    mean = X.mean(keepdims=True)
    std = X.std(keepdims=True)
    X = (X - mean) / std
    print("Data Normalized...")
    return X

# Rescale Data to be in range [rangeMin, rangeMax]
def rescale(X, rangeMin=-1, rangeMax=+1):
    maxi = X.max()
    mini = X.min()
    X = np.interp(X, (mini, maxi), (rangeMin, rangeMax))
    print("Data Rescaled...")
    return X

