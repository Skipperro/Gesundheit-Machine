import os
import random
import threading

import time

soundcard = 'plughw:CARD=AUDIO,DEV=0'
cards = os.system('aplay -L')
if not soundcard in cards:
    print('Desired soundcard not found! Fallback to default soundcard!')
    soundcard = ''


def blessing(sc):
    try:
        devicestring = ''
        if len(sc) > 1:
            devicestring = ' -D ' + sc
        os.system('aplay activation.wav' + devicestring)
        os.system('aplay ' + './gesundheits/' + random.choice([f for f in os.listdir('./gesundheits/')]) + devicestring)
    except:
        os.system('aplay activation.wav')
        os.system('aplay ' + './gesundheits/' + random.choice([f for f in os.listdir('./gesundheits/')]))

def blessingasync(sc):
    th = threading.Thread(target=blessing, args=[sc])
    th.start()

blessingasync(soundcard)