import os
import random
import threading
import simpleaudio as sa
import time

def blessing():
    try:
        os.system('play activation2.wav - q')
        time.sleep(0.5)
        os.system('play ' + './gesundheits/' + random.choice([f for f in os.listdir('./gesundheits/')]) + ' -q')
    except:
        print('PROBLEM WITH SOUND!')

def blessingasync():
    th = threading.Thread(target=blessing)
    th.start()

def bootsound():
    try:
        wave_obj = sa.WaveObject.from_wave_file('activation2.wav')
        wave_obj.play()
        #os.popen('play activation2.wav -q')
        time.sleep(0.3)
        wave_obj.play()
        #os.popen('play activation2.wav -q')
        time.sleep(0.3)
        wave_obj.play()
        #os.popen('play activation2.wav -q')
        time.sleep(3.0)
    except:
        print('PROBLEM WITH SOUND!')

time.sleep(2.0)

bootsound()