# Load various miscillaneous imports
import pandas as pd
import tensforflow as tf
import seaborn as sns
import sys
import os
%pip install scikit-plot
import scikitplot as skplt
import librosa  #Comment this if you don't generate features.
from librosa import display
from datetime import datetime
import pickle
import time
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib import cm
from scipy import signal
from google.colab import files

#Import all necessary tensorflow libraries for CNN, LSTM
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Input, Dropout, Activation, Flatten, Embedding, LSTM, TimeDistributed, MaxPooling1D, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalAveragePooling1D, Convolution2D, Conv1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint

#Import BP-MLL loss
%pip install bpmll
from bpmll import bp_mll_loss

#Next, we need too import the necessary external files for data preperation. If we are using a pre-established set of MFCC features, the files include:

#1.   Dataset Metadata File Names
#2.   Dataset Classes File Names

#1.   MFCC Feature Set

#If we are developing our own MFCC features, exclude 3. from the upload

files.upload()

'''
The purpose of this function is to extract all of the features from the file in question
The librosa library loads the file in question using the resampling type as the maximum quality
Next, we extract the features from the audio sample using the librosa featuresmfcc library. MFCC stands for Mel-Frequency Cepstral Coefficients. These coeffecients are values that collectively create  the **mel-frequency cepstrum **
MFC are a short-term sample of the power levels of a sound. It is based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency. This scale was created in order to perveive equivalent distance between sound frequencies, regardless of the value. This helps us distinguish frequencies that humans percevie to be unnoticable in the lower end of the spectrum. 
To derive the coefficients (MFCC) of the Mel scale, we conduct the following steps:

1. Take the Fourier transform of a time-series signal using a FFT.
2. Map the powers of the spectrum obtained above onto the mel scale, using triangular overlapping windows.
3. Take the  natural logs of the powers at each of the mel frequencies.
4. Transform the mel frequencies using the discrete cosine transform to represent a signal.
5. The MFCC are then the amplitudes of the resulting spectrum

The MFCC have a sweet spot for the number of coefficients.
'''


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

#The metadata shown in the loaded csv below contrains the file names for all of the audio samples that will be used in this experiment.

# Set the path to the full UrbanSound dataset
fulldatasetpath = '/Users/khosrap/Desktop/audio_and_txt_files' # You don't need this if you don't generate feature from audio.
metadata = pd.read_csv('c_w_bothcw_p_h.csv') # assign csv with respiratory file names as the metadata

df = pd.DataFrame(metadata) # converting the metadata to a pandas datframe
df.head() #quickly visualizing our dataframe

## *Optional* Generating MFCC Features
#The block below will generate the MFCC features if they have not been imported and will load them if they have been imported

fea_generation_Flag = False  # To generate features set fea_generation_Flag as True OR reload generated features, i.e. set as False.

if fea_generation_Flag:

    print('**Generating MFCC features ... (This may cost minutes. Please wait.)')

    features = []
    feature_data = []
    class_labels = []

    # Iterate through each sound file from csv and extract the features
    for index, row in metadata.iterrows():
        # the path defined in the block above, make sure this path is correct
        file_name = os.path.join(os.path.abspath(fulldatasetpath),
                                 str(row["fold"]),
                                 str(row["slice_file_name"]))

        class_label = row["classID"]  # grabbing the labeled classes of each respiratory sample
        data = extract_features(file_name)  # generate the MFCC features for the respective file in question

        feature_data.append(data)  # add the MFCCs into the data array
        class_labels.append(class_label)  # add the class_labels to its corresponding array

        features.append([data, class_label])  # combining the two arrays into features array

    with open("feature_data_c_w_bothcw_p_h.txt", "wb") as fp:  # save MFCCs[40,max_pad_len=174]
        pickle.dump(feature_data, fp)

    with open("class_label_c_w_bothcw_p_h.txt", "wb") as fp:  # save labels
        pickle.dump(class_labels, fp)

    # Convert into a panda dataframe
    featuresdf = pd.DataFrame(features, columns=['feature', 'class_label'])

    print('Finished feature extraction from ', len(featuresdf), ' files')
    # Convert features and corresponding classification labels into numpy arrays
    X = np.array(featuresdf.feature.tolist())  # Shape: [8732, 40, 174]
    y = np.array(featuresdf.class_label.tolist())  # Shape: [8732,]

# case if the MFCC features are imported
else:

    print('**Loading .txt MFCC features ...')

    # set the features as the x value
    with open('feature_data_c_w_bothcw_p_h.txt', 'rb') as f:
        x = pickle.load(f)
        X = np.array(x)

    # set the class labels as the y value
    with open('class_label_c_w_bothcw_p_h.txt', 'rb') as f:
        y = pickle.load(f)
        y = np.array(y)

# **Random Forest Classifier**

# Encode the classification labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))
'''
    Mutually exclusive categories classification (multi-class):
        one-hot encoding, last-layer with softmax, categorical_crossentropy loss;
        validation metric: accuracy.
    Multi-label categories classification (multi-label):
        multi-label encoding, last-layer with sigmoid, binary_crossentropy loss;
        validation metric: categorical accuracy.
'''

#transforming the multi-class labels to a multi-label scenario for the "bothCW" label
if True:
    for i in range(len(yy)):
        # if both crackles and wheezes are present
        if y[i] == 2:
            yy[i] = [1, 1, 0, 0, 0]

# split the dataset
x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.20, random_state = 42)

num_rows = 40 # assigning a number of MFCC features as our x shape
num_columns = max_pad_len # assigning the length of each MFCC feature as our quantity of columns (max pad length set to 174)
num_channels = 1 # number of channels (only necessary for CNN and LSTM)

# reshaping the MFCC feature data, both training and testing before using them in models
x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels) #(6985, 40, 174, 1)-->(6985, 40, 174, 1)
x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)

# **LSTM**

#For this LSTM structure we define our timesteps and quantity of sequences before we construct our model

data_dim = x_train.shape[1] # assign the data dimension of a variable
timesteps = x_train.shape[2] # assigning timesteps for the LSTM method
numberofSequence = x_train.shape[0]

### CNN-RNN Model

def CNN_RNN_model(num_rows,num_columns, timesteps):

    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.00, amsgrad=False)
    model = Sequential()

    # For standalone LSTM model comment out all model steps until the LSTM steps
    # For CNN-LSTM model, CNN model components are converted to a Time-Distributed version
    model.add(TimeDistributed(Conv1D(filters=16*2, kernel_size=2, input_shape=(num_rows, num_columns, num_channels,1), activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Dropout(0.2)))

    model.add(TimeDistributed(Conv1D(filters=32*2, kernel_size=2, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Dropout(0.2)))

    model.add(TimeDistributed(Conv1D(filters=64*2, kernel_size=2, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Dropout(0.2)))

    model.add(TimeDistributed(Conv1D(filters=128*2, kernel_size=2, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Dropout(0.2)))
    model.add(TimeDistributed(GlobalAveragePooling1D()))
    model.add(TimeDistributed(Flatten()))

    model.add(LSTM(120, return_sequences=True,input_shape=(num_columns,timesteps)))
    model.add(LSTM(120, return_sequences=False))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

model = CNN_RNN_model(num_rows=num_rows,num_columns=num_columns, timesteps=timesteps)
stats = model.fit(x_train, y_train, batch_size=100,epochs=50, validation_data=(x_test, y_test), verbose=1)
model.summary()

plt.figure(figsize = (10,5))
plt.subplot(1,2,1)
plt.title('Accuracy')
plt.plot(stats.history['accuracy'], label = 'training acc')
plt.plot(stats.history['val_accuracy'], label = 'validation acc')
plt.legend()
plt.subplot(1,2,2)
plt.plot(stats.history['loss'], label = 'training loss')
plt.plot(stats.history['val_loss'], label = 'validation loss')
plt.legend()
plt.title('Loss')
plt.show()
# Evaluating the model on the training and testing set
score = model.evaluate(x_train, y_train, verbose=0)
print("Training Accuracy: ", score[1])

score = model.evaluate(x_test, y_test, verbose=0)
print("Testing Accuracy: ", score[1])

# F-score, and Confusion matrix
predictions = model.predict(x_test)
predictions = np.argmax(predictions, axis = 1)
labels = np.argmax(y_test, axis = 1)

print(classification_report(labels, predictions, target_names = ['crackles','wheezes','pneumoniaNoCW','healthyNoCW']))
print('----------------------------------------------------------')
print(confusion_matrix(labels, predictions))
print('----------------------------------------')
print('Finished.')
