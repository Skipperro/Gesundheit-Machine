import os
import random
import threading

import time

soundcard = 'plughw:CARD=AUDIO,DEV=0'
cards = os.popen('aplay -L').read()
if str(cards).find(soundcard) < 0:
    print('Desired soundcard not found! Fallback to default soundcard!')
    soundcard = ''


def blessing(sc):
    try:
        devicestring = ''
        if len(sc) > 1:
            devicestring = ' -D ' + sc
        os.popen('aplay activation2.wav' + devicestring)
        time.sleep(0.5)
        os.system('aplay ' + './gesundheits/' + random.choice([f for f in os.listdir('./gesundheits/')]) + devicestring)
    except:
        print('PROBLEM WITH SOUND!')

def blessingasync(sc):
    th = threading.Thread(target=blessing, args=[sc])
    th.start()

def bootsound(sc):
    try:
        devicestring = ''
        if len(sc) > 1:
            devicestring = ' -D ' + sc
        os.popen('aplay activation2.wav' + devicestring)
        time.sleep(0.3)
        os.popen('aplay activation2.wav' + devicestring)
        time.sleep(0.3)
        os.popen('aplay activation2.wav' + devicestring)
        time.sleep(3.0)
    except:
        print('PROBLEM WITH SOUND!')

time.sleep(2.0)

bootsound(soundcard)