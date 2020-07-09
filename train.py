from scipy.io.wavfile import read, write
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.keras import regularizers
import time
import threading
asd = tf.test.is_gpu_available()
random.seed(7)

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Reads audio file and outputs it as numpy array normalized in range from -1 to 1
def get_wave_normalized(file_path):
    wav = read(file_path)
    if len(wav[1].shape) > 1:
        arr = np.array(wav[1], dtype='float32')[:, 1]
    else:
        arr = np.array(wav[1], dtype='float32')
    arr = normalize_array(arr)
    arr = np.reshape(arr, (len(arr), 1))
    return arr


# Normalizes array in range from -1 to 1
def normalize_array(arr):
    return arr / (np.max(np.abs(arr)))


# Generates dataset for training or testing.
# Count - Integer number of wav files to generate.
# Sneeze - Boolean value if the wav files should be generated with added sneeze.
# Test - Boolean value if the dataset should be for testing or training.
def generate_dataset(count, with_sneeze, test=False, save_files=True):
    backgrounds = []
    sneezes = []
    batch = []
    #positives = []
    #negatives = []
    for f in os.listdir('./dataset/backgrounds'):
        backgrounds.append(get_wave_normalized('./dataset/backgrounds/' + f))
    if with_sneeze:
        for f in os.listdir('./dataset/sneezes'):
            sneezes.append(get_wave_normalized('./dataset/sneezes/' + f))


    for i in range(count):

        bg = random.choice(backgrounds)
        bg_sample_start = random.randint(0, bg.shape[0] - 44100)
        bg_sample_stop = bg_sample_start + 44100
        if with_sneeze:
            bg_sample = bg[bg_sample_start:bg_sample_stop] * random.uniform(0.0, 0.6)
        else:
            bg_sample = bg[bg_sample_start:bg_sample_stop]
        sneeze_sample = np.zeros(200000)
        sneeze_sample = np.reshape(sneeze_sample, (200000,1))
        if with_sneeze:
            if test:
                sneeze_wave = random.choice(sneezes[:int(len(sneezes)/2)])
            else:
                sneeze_wave = random.choice(sneezes[int(len(sneezes)/2):])
            shift = 0 #random.randint(0, 10000)
            sneeze_sample[shift:shift+sneeze_wave.shape[0]] = sneeze_wave
        sneeze_sample = sneeze_sample[0:44100]
        final_sample = normalize_array(bg_sample + sneeze_sample)
        final_sample_int = np.array(final_sample * 32768, dtype='int16')
        dir_name = 'train'
        if test:
            dir_name = 'test'
        group_name = '0'
        if with_sneeze:
            group_name = '1'
        if save_files:
            write('./dataset/' + dir_name + '/' + group_name + '/' + str(i) + '.wav', 44100, final_sample_int)
            # print('Created file: ./dataset/' + dir_name + '/' + group_name + '/' + str(i) + '.wav')
        batch.append(final_sample)
    return batch


def dataset_generator(batch_size=32, test=False):
    negatives = []
    positives = []
    if test:
        for f in os.listdir('./dataset/test/0'):
            negatives.append('./dataset/test/0/' + f)
        for f in os.listdir('./dataset/test/1'):
            positives.append('./dataset/test/1/' + f)
    else:
        for f in os.listdir('./dataset/train/0'):
            negatives.append('./dataset/train/0/' + f)
        for f in os.listdir('./dataset/train/1'):
            positives.append('./dataset/train/1/' + f)
    while True:
        batch = []
        ground_truth = []
        for i in range(batch_size):
            if random.random() > 0.5:
                batch.append(get_wave_normalized(random.choice(positives)))
                ground_truth.append([1])
            else:
                batch.append(get_wave_normalized(random.choice(negatives)))
                ground_truth.append([0])
        batch = np.array(batch, dtype='float32')
        ground_truth = np.array(ground_truth, dtype='int8')
        yield batch, ground_truth


def endless_batch_generator(batch_size=64):
    while True:
        positives = np.array(generate_dataset(int(batch_size/2), True, save_files=True), dtype='float32')
        negatives = np.array(generate_dataset(int(batch_size/2), False, save_files=True), dtype='float32')
        x = np.concatenate((positives, negatives))
        y = np.concatenate((np.ones((int(batch_size/2),1), dtype='int8'), np.zeros((int(batch_size/2),1), dtype='int8')))
        yield x, y


def endless_background_generator(batch_size=64):
    global background_X, background_Y, tlock
    while background_X.shape[0] < batch_size * 2:
        time.sleep(1)

    while True:
        tlock.acquire()
        startindex = random.randint(0, background_X.shape[0] - batch_size - 1)
        x = np.array(background_X[startindex:startindex + batch_size])
        y = np.array(background_Y[startindex:startindex + batch_size])
        tlock.release()
        yield x, y


def background_dataset_generator(max_size=16000):
    global background_X, background_Y, tlock
    batch_size = 128
    while True:
        positives = np.array(generate_dataset(int(batch_size / 2), True, save_files=False), dtype='float32')
        negatives = np.array(generate_dataset(int(batch_size / 2), False, save_files=False), dtype='float32')
        x = np.concatenate((positives, negatives))
        y = np.concatenate((np.ones((int(batch_size / 2), 1), dtype='int8'), np.zeros((int(batch_size / 2), 1), dtype='int8')))

        tlock.acquire()
        background_X = np.concatenate((background_X, x), 0)
        background_Y = np.concatenate((background_Y, y), 0)
        if background_X.shape[0] >= max_size:
            background_X = background_X[batch_size:]
            background_Y = background_Y[batch_size:]
        tlock.release()


background_X = np.zeros((1, 44100, 1), dtype='float32')
background_Y = np.zeros((1, 1), dtype='int8')
tlock = threading.Lock()

bthread = threading.Thread(target=background_dataset_generator)
#bthread.start()



# Create train dataset
#generate_dataset(8192, True, False)
#generate_dataset(8192, False, False)

# Create test dataset
#generate_dataset(1024, True, True)
#generate_dataset(1024, False, True)


# Create model
actf = 'tanh'
drop = 0.1
neurons = 32
model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(44100, 1)))
model.add(tf.keras.layers.Conv1D(neurons, 4, strides=4, activation=actf,
    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5)))
model.add(tf.keras.layers.Dropout(drop))
model.add(tf.keras.layers.MaxPool1D(4))
model.add(tf.keras.layers.Conv1D(neurons * 2, 8, strides=4, activation=actf,
    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5)))
model.add(tf.keras.layers.Dropout(drop))
model.add(tf.keras.layers.MaxPool1D(4))
model.add(tf.keras.layers.Conv1D(neurons * 4, 16, strides=4, activation=actf,
    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5)))
model.add(tf.keras.layers.Dropout(drop))
model.add(tf.keras.layers.GlobalMaxPool1D())
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

print(model.summary())
#exit()
model.compile(optimizer=tf.keras.optimizers.Adadelta(0.01), loss='mse', metrics=['acc'])
try:
    model.load_weights('model.h5', True, True)
    pass
except:
    print('Unable to load weights')
    #exit()
valdata = dataset_generator(6400, True).__next__()
for i in range(100):
    model.fit_generator(dataset_generator(512), 512, 10, validation_data=valdata)
    #, callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    tf.keras.models.save_model(model, 'model.h5')
    model.save_weights('model.weights')
    print("Saved model to disk")

#testdata = dataset_generator(32).__next__()
#model = ak.ImageClassifier(max_trials=20)
#model.fit(testdata[0], testdata[1], epochs=5, validation_split=0.5)
#model = model.export_model()
print(model.summary())

#model.fit_generator(dataset_generator(32), 100, 10)
testdata = dataset_generator(10).__next__()
preds = model.predict(testdata[0])
print(testdata[1])
print(preds)

valpreds = model.evaluate(valdata[0], valdata[1], 64)
print(valpreds)
from pydub import AudioSegment
from pydub.playback import play
while True:
    song = AudioSegment.from_wav("./activation.wav")
    play(song)
    time.sleep(10)