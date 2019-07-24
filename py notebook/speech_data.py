

import os
import re
import sys
import wave

import numpy
import numpy as np
import skimage.io  
import librosa
import matplotlib

from random import shuffle
from six.moves import urllib
from six.moves import xrange 


SOURCE_URL = 'http://pannous.net/files/' 
DATA_DIR = 'data/'
pcm_path = "data/spoken_numbers_pcm/" 
wav_path = "data/spoken_numbers_wav/" 
path = pcm_path
CHUNK = 4096
test_fraction=0.1 


class Source:  # labels
  DIGIT_WAVES = 'spoken_numbers_pcm.tar'
  DIGIT_SPECTROS = 'spoken_numbers_spectros_64x64.tar'  
  NUMBER_WAVES = 'spoken_numbers_wav.tar'
  NUMBER_IMAGES = 'spoken_numbers.tar'  # width=256 height=256
  WORD_SPECTROS = 'https://dl.dropboxusercontent.com/u/23615316/spoken_words.tar'  
  TEST_INDEX = 'test_index.txt'
  TRAIN_INDEX = 'train_index.txt'

from enum import Enum
class Target(Enum):  # labels
  digits=1
  speaker=2
  words_per_minute=3
  word_phonemes=4
  word=5\
  sentence=6
  sentiment=7
  first_letter=8

def speaker(file):  
  return file.split("_")[1]

def get_speakers(path=pcm_path):
  files = os.listdir(path)
  def nobad(file):
    return "_" in file and not "." in file.split("_")[1]
  speakers=list(set(map(speaker,filter(nobad,files))))
  print(len(speakers)," speakers: ",speakers)
  return speakers

def load_wav_file(name):
  f = wave.open(name, "rb")
 
  chunk = []
  data0 = f.readframes(CHUNK)
  while data0:  
   
    data = numpy.fromstring(data0, dtype='uint8')
    data = (data + 128) / 255. 
    
    chunk.extend(data)
    data0 = f.readframes(CHUNK)
  :chunk[0:CHUNK * 2]  
  chunk.extend(numpy.zeros(CHUNK * 2 - len(chunk))) 
  
  return chunk




def mfcc_batch_generator(batch_size=10, source=Source.DIGIT_WAVES, target=Target.digits):
  
  if target == Target.speaker: speakers = get_speakers()
  batch_features = []
  labels = []
  files = os.listdir(path)
  while True:
    print("loaded batch of %d files" % len(files))
    shuffle(files)
    for wav in files:
      if not wav.endswith(".wav"): continue
      wave, sr = librosa.load(path+wav, mono=True)
      if target==Target.speaker: label=one_hot_from_item(speaker(wav), speakers)
      elif target==Target.digits:  label=dense_to_one_hot(int(wav[0]),10)
      elif target==Target.first_letter:  label=dense_to_one_hot((ord(wav[0]) - 48) % 32,32)
      else: raise Exception("todo : labels for Target!")
      labels.append(label)
      mfcc = librosa.feature.mfcc(wave, sr)
      
      mfcc=np.pad(mfcc,((0,0),(0,80-len(mfcc[0]))), mode='constant', constant_values=0)
      batch_features.append(np.array(mfcc))
      if len(batch_features) >= batch_size:
       
        yield batch_features, labels  
        batch_features = []  
        labels = []




