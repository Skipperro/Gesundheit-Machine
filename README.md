![Banner](/images/banner.png)
# Gesundheit-Machine

Allergy season is coming? People sneezing around you all the time? You are forced by some weird social norms to respond with "Bless you / Gesundheit" to every sneeze around you and it distracts you and makes your code buggy?

What if there was an easy way to automate this social interaction and let you focus on what you really want to do?
You can now do this with the state-of-the-art, AI-powered Gesundheit-Machine, Made-in-Germany, designed specifically for German market.

## How it works?

Gesundheit-Machine will sit on your desk and listen, just like Amazon Alexa, Google Home or Siri in your iPhone. If someone will sneeze nearby, Gesundheit-Machine will detect it and automagically respond with one of pre-recorded message. You don't have to do anything. You can even record your own answers with your voice so people will think that you actually wish them well yourself! It will work even if you are not there, so no sneeze will be left unresponded.

## But seriously... how it technically works?

It's a piece of software written in Python that uses Deep Neural Network created in Tensorflow/Keras. The neural network is trained to classify 1 second audio sample to tell if it sounds like a sneeze or not. 

The software constantly reads audio input from microphone and every few milliseconds sends most recent 1-second audio sample to neural network for classification. The network classifies each sample and decides if it sounds like a sneeze or not. If few samples in a row are positively classified a response sequence is triggered.

The software is running on simple Raspberry Pi with microphone and speaker connected to it.

## Was it hard to develop?

Yes!

Initially I thought it will be easy, just take any MIT licensed voice-to-text neural network and retrain it to detect sneeze, but it was way harder than this.

### Problem 1 - Dataset

There is no publicly available dataset of sneezes on the Internet. I had to make my own.

Luckly, for some weird reason, there is many compilations of people sneezing on YouTube. Don't know why someone would watch this stuff, but for me it was a base on which I could build my dataset. I've downloaded audio of every "sneeze" YouTube video I could find and manually extracted those 1 second parts where someone sneeze. This was my dataset of extracted sneezes but this was only the first step.

For training I've also needed some examples of audio samples that are not sneeze. To generate them I've downloaded many other YouTube videos with ambient noise like office space, construction zone, party music, pub conversations, screaming kids ect. and wrote a code that would randomly cut those audio into 1-second samples.

To further improve detection accuracy I've also started to combine non-sneeze background sounds with sneeze sample overlayed on top of it, so the AI could clearly see the difference between the two.

But even this dataset is not perfect. For example, there is a VAST overrepresentation of female sneezes, because for some weird reason there are almost only "sneezing girls" compilations on YouTube and almost none of "boys", so naturally my AI is better on recognizing if a female sneeze than if a male sneeze.

I've also figured out, that it's better to miss some sneezes rather than trigger false-positive reaction when noone sneezed. To prevent false-positives I've created a first version of detection AI, set parameters to be very sensitive and let it listen... for days. Each time it thought it heard sneeze this 1-second audio sample was saved. After few days of listening what happens at my home and office I had thousends of detected sneezes that were false-positive, so I've added those to the dataset as well (as non-sneeze samples).

I have now over 130.000 unique samples of 1-second audio, about 30% of them with sneeze, occupying about 12 GB of space. I can also mix those samples freely to create more examples as there are atoms in the universe.

At this point I was pretty convinced, that I own a best sneeze-detection dataset on the planet. :)

### Problem 2 - Not all sneezes are equal

If you want to detect some trigger word like "Alexa", "OK Google" or "Hey Siri", it's way easier that detecting sneeze. People have different voices, but the overall "Trigger-word" melody is roughly the same for everyone. During development of this software I found out, that it is not the case for sneezing. There is as many types of sneeze as there are people. You can have silent squeeks or loud roars, discrete cough or bursting explosion, single or multiple tones. Also the signal itself is much less distinct and shorter than the classical "Trigger-words" used by current AI Assistants.

That's why there are still some false-positives I simply cannot erradicate from my algorithm, because they are too similar to real sneeze.

Some examples:

* Saying "Oh Cool!" - hard to manually classify even for me based only on 1-second audio sample.
* Saying "Albert" or "Apple" the right way - too close to "A-Choo".
* Banging spoon on a plate - erradicating this pattern vastly reduces detection rate.
* Screaming kids - too similar to many female sneeze examples.
* Dropping microphone - too similar to sneeze directly into the mic.
* Blowing into the microphone or rubbing it - same problem.
* Laughing - some people just laugh the "sneezy" way.

### Problem 3 - AI have some hardware requirements

Initially I've simply created bigger and bigger neural networks to handle all the subtle edge-cases until I've finally was able to achieve 99,97% accuracy in validation dataset and had almost perfect sneeze-detection model. Problem is - I needed to detect sneezes in realtime. Using this big model required something like RTX 2060 GPU running constantly at full power. Raspberry Pi can provide only small fraction of this performance, so I had to optimize the neural architecture big way.

Over the course of few weeks I've created and trained over 200 different networks, one after another, to test which changes will give most performance improvement with minimal accuracy reduction.

The optimal network I found can run smoothly on Raspberry Pi and have still decent 93% accuracy. It is optimized for low CPU utilization at the cost of high RAM usage. It can check over 10 samples per second while keeping Raspberry under 80°C.

To avoid false positives from a single check I've added additional threashold - one positive detection still don't trigger the response. To trigger "Gesundheit" a few checks in a row (usually over the span of 500 ms) must be classified as positive.

With this workaround even Raspberry Pi provides satysfying results.

## How to build your own Gesundheit-Machine?

### What you will need

* Raspberry Pi (3 or better) with SD card. I recommend also some chassis.
* One Raspberry-compatible microphone (I've used old webcam with microphone).
* One Raspberry-compatible speaker.
* Some mouse, keyboard and HDMI display for installation. 
* Code from my GitHub repository.

### Building the hardware

Simply put Raspberry Pi in chassis, connect microphone and speaker to it then install some Linux based OS (ideally one using systemd to allow installation of software as a service).

### Install Gesundheit-Machine software

* Install Python3 (I've used version 3.6 but it should work with any 3.X version).
* Clone this repository.
* Install all the python packages from requirements.txt file. Tensorflow 2.2.0 may require manual download from github.
* Run main.py with python. AI model is already included in the repository.

### Install service to run automatically on start

First edit gesundheitmachine.service file.

* Change "User" to proper user name in your system.
* Change "ExecStart" to proper working directory path and python executable path.
* Save/copy this file to `/etc/systemd/system/gesundheitmachine.service`.
* Run `sudo systemctl enable gesundheitmachine` and reboot the system.
* Gesundheit-Machine should be running automatically on the background.

At this point you can disconnect mouse, keyboard and display and deploy the Gesundheit-Machine to it's target location.

### Retriaining the network

If you want to retrain the network yourself it can be complicated.

Because the dataset I've used for training consist of fragments of YouTube videos and recordings of my family, I won't share it at all to avoid any legal problems. 

If you want to create dataset yourself, you can use `train.py` file to train neural network on it, but out-of-the-box retraining the dataset is not possible.

### Add your own voices

If you want to add your own responses simply record them somewhere, save as WAV file and put them into "gesundheits" directory. Each time a sneeze is detected one of the files (chosen randomly) will be played.

## Some statistics

* Total work time: many, many weeks. Much more than I'm willing to admit.
* Dataset used for training: over 130.000 samples of 1 second audio (12 GB in wav format)
* Amount of different neural models tested: over 200.
* Final neural network size: only 620 KB!
* Final training time: about 20 minutes on single Titan Xp GPU.
* Over 200 recorded responses done by two voice actresses.
* Animals hurt in the procces of making this device: 0.

## Plans/ideas for the future

* Optimize it even further.
* Make it work as a mobile app.
* Get Morgan Freeman to record "Bless you" for English version.
* Expand it with a tissue dispenser module.
* Add some LEDs to light-up while the sound is played.
