# Load various imports
import pandas as pd
import os
#import librosa  #Comment this if you don't generate features.

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
model = Sequential()
model.add(Conv2D(filters=16*2, kernel_size=2, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=32*2, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=64*2, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=128*2, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())

model.add(Dense(num_labels, activation='softmax'))
#----------------------------------------------------------------------------------------------#
# Load pretrained softmax-crossEntropyLoss model
if False:
    model.load_weights(filepath='./saved_models_Keras/weights.best.basic_cnn.hdf5')
    print('Loaded Model from disk')
    # Set every layer to be non-trainable:
    for k, v in model._get_trainable_state().items():
        k.trainable = False
#----------------------------------------------------------------------------------------------#
# Print current trainable map
#print(model._get_trainable_state())
#----------------------------------------------------------------------------------------------#
# Plot the model
# Need to install pydot and graphviz
# plot_model(model, show_shapes=True, show_layer_names = True)
#----------------------------------------------------------------------------------------------#
# Compile the model
if False:
    opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.00, amsgrad=False)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt) #optimizer='adam'
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam') #optimizer='adam'
# Display model architecture summary
model.summary()

# Calculate pre-training accuracy
score = model.evaluate(x_test, y_test, verbose=1)
accuracy = 100*score[1]

print("Pre-training accuracy: %.4f%%" % accuracy)
#----------------------------------------------------------------------------------------------#
if False:
    stats = model.fit_generator(generator = train_gen.generate_keras(batch_size),
                                steps_per_epoch = train_gen.n_available_samples() // batch_size,
                                validation_data = test_gen.generate_keras(batch_size),
                                validation_steps = test_gen.n_available_samples() // batch_size,
                                epochs = n_epochs)
#----------------------------------------------------------------------------------------------#
num_epochs = 60
num_batch_size = 128

checkpointer = ModelCheckpoint(filepath='weights.best.basic_cnn.hdf5',
                               verbose=1, save_best_only=True)
start = datetime.now()

stats = model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)
#----------------------------------------------------------------------------------------------#
if False:
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
#----------------------------------------------------------------------------------------------#
# F-score, and Confusion matrix
predictions = model.predict(x_test)
predictions = np.argmax(predictions, axis = 1)
labels = np.argmax(y_test, axis = 1)

print(classification_report(labels, predictions, target_names = ['crackles','wheezes','bothcw','pneumoniaNoCW','healthyNoCW']))
print('----------------------------------------------------------')
print(confusion_matrix(labels, predictions))
'''
               precision    recall  f1-score   support

     crackles       0.74      0.90      0.81       356
      wheezes       0.68      0.68      0.68       170
       bothcw       0.72      0.26      0.38       127
pneumoniaNoCW       0.86      0.77      0.81        47
  healthyNoCW       0.77      0.92      0.84        60

     accuracy                           0.74       760
    macro avg       0.76      0.70      0.70       760
 weighted avg       0.73      0.74      0.71       760

----------------------------------------
[[321  18   5   2  10]
 [ 45 115   8   1   1]
 [ 58  33  33   1   2]      <-----Bo: very bad!!! Need to solve this first.
 [  6   2   0  36   3]
 [  3   0   0   2  55]]
'''
print('----------------------------------------')
print('Finished.')