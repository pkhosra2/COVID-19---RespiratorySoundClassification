# Load various imports
import pandas as pd
import os
import librosa  #Comment this if you don't generate features.

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
''' 
import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
#tensorflow.compat.v1.
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
#from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
'''
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Convolution2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import optimizers
#from keras.utils import np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
#from keras.utils.vis_utils import plot_model
from tensorflow.keras.utils import plot_model

from sklearn import metrics
from datetime import datetime
import pickle
import time
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
#-----------------------------------------------------------------------------------------------#
max_pad_len = 400

def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None

    return mfccs

#----------------------------------------------------------------------------------------------#
# Set the path to the full UrbanSound dataset
fulldatasetpath = 'Path_To_Audios' #You don't need this if you don't generate feature from audio.
metadata = pd.read_csv('c_w_bothcw_p_h.csv')

fea_generation_Flag = False #Generate features(True) OR reload generated features(False).
if fea_generation_Flag:
    print('**Generating MFCC features ... (This may cost minutes. Please wait.)')
    features = []
    feature_data = []
    class_labels =[]
    # Iterate through each sound file and extract the features
    for index, row in metadata.iterrows():
        file_name = os.path.join(os.path.abspath(fulldatasetpath), str(row["fold"]),
                                 str(row["slice_file_name"]))

        class_label = row["classID"]
        data = extract_features(file_name)

        feature_data.append(data)
        class_labels.append(class_label)

        features.append([data, class_label])

    with open("feature_data_c_w_bothcw_p_h.txt", "wb") as fp: # save MFCCs[40,max_pad_len=174]
        pickle.dump(feature_data, fp)
    with open("class_label_c_w_bothcw_p_h.txt", "wb") as fp:  # save labels
        pickle.dump(class_labels, fp)

    # Convert into a Panda dataframe
    featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])

    print('Finished feature extraction from ', len(featuresdf), ' files')
    # Convert features and corresponding classification labels into numpy arrays
    X = np.array(featuresdf.feature.tolist())  # Shape: [8732, 40, 174]
    y = np.array(featuresdf.class_label.tolist())  # Shape: [8732,]
else:
    print('**Loading .txt MFCC features ...')
    with open('feature_data_c_w_bothcw_p_h.txt', 'rb') as f:
        x = pickle.load(f)
        X = np.array(x)
    with open('class_label_c_w_bothcw_p_h.txt', 'rb') as f:
        y = pickle.load(f)
        y = np.array(y)
#----------------------------------------------------------------------------------------------#
# Encode the classification labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))

# split the dataset
x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)
#----------------------------------------------------------------------------------------------#
num_rows = 40
num_columns = max_pad_len
num_channels = 1

x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels) #(6985, 40, 174, 1)-->(6985, 40, 174, 1)
x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)   #(1747, 40, 174)-->(1747, 40, 174, 1)

num_labels = yy.shape[1]
#filter_size = 2

# Construct model
model_pretrained = Sequential()
model_pretrained.add(Conv2D(filters=16*2, kernel_size=2, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
model_pretrained.add(MaxPooling2D(pool_size=2))
model_pretrained.add(Dropout(0.2))

model_pretrained.add(Conv2D(filters=32*2, kernel_size=2, activation='relu'))
model_pretrained.add(MaxPooling2D(pool_size=2))
model_pretrained.add(Dropout(0.2))

model_pretrained.add(Conv2D(filters=64*2, kernel_size=2, activation='relu'))
model_pretrained.add(MaxPooling2D(pool_size=2))
model_pretrained.add(Dropout(0.2))

model_pretrained.add(Conv2D(filters=128*2, kernel_size=2, activation='relu'))
model_pretrained.add(MaxPooling2D(pool_size=2))
model_pretrained.add(Dropout(0.2))
model_pretrained.add(GlobalAveragePooling2D())

model_pretrained.add(Dense(num_labels, activation='softmax'))
#----------------------------------------------------------------------------------------------#
# Load pretrained softmax-crossEntropyLoss model
model_pretrained.load_weights(filepath='weights.best.basic_cnn.hdf5')
print('Loaded Model from disk')

# Print current trainable map
#print(model_pretrained._get_trainable_state())

# Set every layer to be non-trainable:
for k,v in model_pretrained._get_trainable_state().items():
    k.trainable = False
#----------------------------------------------------------------------------------------------#
# Construct model
model = Sequential()
model.add(model_pretrained)
#----------------------------------------------------------------------------------------------#
# --> Plot the model
# --> Need to install pydot and graphviz
#plot_model(model, show_shapes=True, show_layer_names = True)
#----------------------------------------------------------------------------------------------#
# --> Predict individual sounds.
def print_prediction(file_name, label):
    ground_label = {0: 'Crackles', 1: 'Wheezes', 2: 'BothC&W', 3: 'PneumoniaNoC&W', 4: 'HealthyNoC&W'}
    prediction_feature = extract_features(file_name)
    prediction_feature = prediction_feature.reshape(1, num_rows, num_columns, num_channels)

    predicted_vector = model.predict_classes(prediction_feature)
    predicted_class = le.inverse_transform(predicted_vector)
    print("The predicted class is:", predicted_class[0], '\n')

    predicted_proba_vector = model.predict_proba(prediction_feature)
    predicted_proba = predicted_proba_vector[0]
    for i in range(len(predicted_proba)):
        category = le.inverse_transform(np.array([i]))
        print(category[0], "\t", format(ground_label[category[0]], '15s'), "\t\t : ", format(predicted_proba[i], '.32f'))
    print('\nTrue label: ', label, "\t", ground_label[label])


# Class: bothcw
# ground label {0: 'Crackles', 1: 'Wheezes', 2: 'BothC&W', 3: 'PneumoniaNoC&W', 4: 'HealthyNoC&W'}
filename1 = 'bothcw_1_107_3p2_Al_mc_AKGC417L.wav'
print_prediction(filename1, label=2)
filename2 = 'bothcw_1_110_1p1_Al_sc_Meditron.wav'
print_prediction(filename2, label=2)


