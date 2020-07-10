#!./venv/bin python3

import numpy as np
import sounddevice as sd
import time
import argparse
import queue
import sys
import os
from pydub import AudioSegment
from pydub.playback import play
from scipy.io.wavfile import read, write
from tensorflow.keras import regularizers
import tensorflow as tf
import uuid
import random

# Uncomment to disable GPU support
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


# Create model
try:
    model = tf.keras.models.load_model('model.keras')
except Exception as e:
    print(e)
    exit()

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    'channels', type=int, default=[1], nargs='*', metavar='CHANNEL',
    help='input channels to plot (default: the first)')
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-w', '--window', type=float, default=1000, metavar='DURATION',
    help='visible time slot (default: %(default)s ms)')
parser.add_argument(
    '-i', '--interval', type=float, default=200,
    help='minimum time between plot updates (default: %(default)s ms)')
parser.add_argument(
    '-b', '--blocksize', type=int, help='block size (in samples)')
parser.add_argument(
    '-r', '--samplerate', type=float, help='sampling rate of audio device')
parser.add_argument(
    '-n', '--downsample', type=int, default=1, metavar='N',
    help='display every Nth sample (default: %(default)s)')
parser.add_argument(
    '-c', '--capture', type=int, default=0, metavar='N',
    help='capture positive sounds (default: %(default)s)')
parser.add_argument(
    '-t', '--threshold', type=int, default=60,
    help='threshold for counting as positive (default: %(default)s)')
parser.add_argument(
    '-s', '--samples', type=int, default=2,
    help='threshold for positive samples to trigger response (default: %(default)s)')
args = parser.parse_args(remaining)
if any(c < 1 for c in args.channels):
    parser.error('argument CHANNEL: must be >= 1')
mapping = [c - 1 for c in args.channels]  # Channel numbers start with 1
q = queue.Queue()


def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    # Fancy indexing with mapping creates a (necessary!) copy:
    q.put(indata[::args.downsample, mapping])


sneeze_count = 0
last_sneeze = time.time()
last_capture = time.time()

os.system('clear')
print('Gesundheit Maschine - Listening...')


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

X = []
last_ai_check = time.time()
last_silence = time.time()

def update_plot(frame):
    global X
    global plotdata
    global sneeze_count
    global last_sneeze
    global last_capture
    global last_ai_check
    global last_silence

    data = np.empty((0,1), 'float32')
    # Get the new data
    while len(data) < 20000:
        try:
            newdata = q.get(block=True, timeout=0.2) # get_nowait()
            data = np.concatenate((data, newdata), axis=0) # data + newdata
        except queue.Empty:
            break

    while len(data) >= 4410:
        #print(len(data))
        if len(data) >= 4410:
            shift = 4410
            plotdata = np.roll(plotdata, -shift, axis=0)
            plotdata[-shift:, :] = data[:shift]
            data = data[shift:]
        #else:
         #   if len(data) > 0:
          #      shift = len(data)
           #     plotdata = np.roll(plotdata, -shift, axis=0)
            #    plotdata[-shift:, :] = data[:shift]
             #   data=[]

        if time.time() - last_sneeze > 3.0 and plotdata.max() > 0.1:
            X.append(normalize_array(np.array(plotdata)))

            if len(data) <= 4410 and time.time() - last_ai_check > 0.5:
                last_ai_check = time.time()
                X = np.array(X)
                #X = normalize_array(X)
                start = time.time()
                preds = model.predict(X)
                predtimestr = '{:4d}'.format(int((time.time() - start)*1000)) + ' ms'
                X = []
                maxprob = 0.0

                for i in range(len(preds)):
                    prob = preds[i][0]
                    if prob > maxprob:
                        maxprob = prob
                    if time.time() - last_sneeze > 5.0 and prob > (args.threshold / 100.0):
                        sneeze_count += 1

                        if sneeze_count > args.samples:
                            print('SNEEZE DETECTED!!')
                            if args.capture:
                                last_capture = time.time()
                                allcaptures = os.listdir('./captures/')
                                if allcaptures.__len__() > 1000:
                                    os.remove('./captures/' + str(random.choice(allcaptures)))
                                write('./captures/' + str(uuid.uuid4()) + '.wav', 44100,
                                      np.array(X[0] * 32000, dtype='int16'))
                                #continue
                            song = AudioSegment.from_wav("./activation.wav")
                            play(song)
                            song = AudioSegment.from_wav(
                                './gesundheits/' + random.choice([f for f in os.listdir('./gesundheits/')]))
                            play(song)
                            last_sneeze = time.time()
                    else:
                        if sneeze_count > 0:
                            sneeze_count = 0
                print('Probability: ' + '{:3d}'.format(int(maxprob*100)) + '%   Samples per batch: ' + '{:2d}'.format(len(preds)) + '   Batch time: ' + predtimestr)
        else:
            if time.time() - last_silence > 1.0:
                print('Silence...')
                last_silence = time.time()
            X = []
            if sneeze_count > 0:
                sneeze_count = 0

    #for column, line in enumerate(lines):
    #    line.set_ydata(plotdata[:, column])
    #return lines


try:
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, 'input')
        args.samplerate = device_info['default_samplerate']

    length = int(44100 * (args.window / 1000))
    plotdata = np.zeros((length, len(args.channels)))



    # Blocksize=44100 to prevent input overflow on Raspberry Pi.
    stream = sd.InputStream(
        device=args.device, blocksize=44100, channels=max(args.channels),
        samplerate=args.samplerate, callback=audio_callback)
    #ani = FuncAnimation(fig, update_plot, interval=args.interval, blit=True)

    with stream:
        while True:
            time.sleep(0.1)
            update_plot(1)
        #plt.show()
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))