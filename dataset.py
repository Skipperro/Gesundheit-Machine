from scipy.io.wavfile import read, write
import numpy as np
import os
import random

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

def add_random_noise(arr, magnitude):
    noise = np.random.normal(0, magnitude*random.random(), arr.shape)
    new_signal = arr + noise
    return new_signal

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
            shift = random.randint(0, 10000)
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
            print('Created file: ./dataset/' + dir_name + '/' + group_name + '/' + str(i) + '.wav')


generate_dataset(64000, True, False)
generate_dataset(1600, True, True)
generate_dataset(64000, False, False)
generate_dataset(1600, False, True)