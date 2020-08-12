# coding: utf-8

# # Mozilla TTS on CPU Real-Time Speech Synthesis 

# We use Tacotron2 and MultiBand-Melgan models and LJSpeech dataset.
# 
# Tacotron2 is trained using [Double Decoder Consistency](https://erogol.com/solving-attention-problems-of-tts-models-with-double-decoder-consistency/) (DDC) only for 130K steps (3 days) with a single GPU.
# 
# MultiBand-Melgan is trained  1.45M steps with real spectrograms.
# 
# Note that both model performances can be improved with more training.

# ### Download Models

# In[1]:


#!gdown --id 1dntzjWFg7ufWaTaFy80nRz-Tu02xWZos -O tts_model.pth.tar
#!gdown --id 18CQ6G6tBEOfvCHlPqP8EBI4xWbrr9dBc -O config.json


# In[2]:


#!gdown --id 1Ty5DZdOc0F7OTGj9oJThYbL5iVu_2G0K -O vocoder_model.pth.tar
#!gdown --id 1Rd0R_nRCrbjEdpOwq6XwZAktvugiBvmu -O config_vocoder.json
#!gdown --id 11oY3Tv0kQtxK_JPgxrfesa99maVXHNxU -O scale_stats.npy


# ### Setup Libraries

# In[3]:


#!sudo apt-get install espeak ffmpeg -y


# In[4]:


#!git clone https://github.com/mozilla/TTS


# In[5]:


#%cd TTS
#!git checkout b1935c97
#!pip install -r requirements.txt
#!python setup.py install
#!pip install inflect pydub
#%cd ..


# ### Load Models

# In[6]:


import gc
import copy
import os
import torch
import time
import IPython
import numpy as np
import scipy.io.wavfile
import math
#from playsound import playsound

from TTS.utils.generic_utils import setup_model
from TTS.utils.io import load_config
from TTS.utils.text.symbols import symbols, phonemes
from TTS.utils.audio import AudioProcessor
from TTS.utils.synthesis import synthesis
from TTS.vocoder.utils.generic_utils import setup_generator


# In[7]:


import resource
#TTS Class
class TTSModel:
    def __init__(self, TTS_MODEL, TTS_CONFIG, VOCODER_MODEL, VOCODER_CONFIG, use_cuda, use_gl):
        self.use_cuda = use_cuda
        self.use_gl = use_gl 
        # model paths
        self.tts_config = load_config(TTS_CONFIG)
        vocoder_config = load_config(VOCODER_CONFIG)
        # load audio processor
        self.ap = AudioProcessor(**self.tts_config.audio)
        # LOAD TTS MODEL
        # multi speaker 
        self.speaker_id = None
        speakers = []
        # load the model
        num_chars = len(phonemes) if self.tts_config.use_phonemes else len(symbols)
        self.model = setup_model(num_chars, len(speakers), self.tts_config)
        # load model state
        self.cp =  torch.load(TTS_MODEL, map_location=torch.device('cpu'))
        # load the model
        self.model.load_state_dict(self.cp['model'])
        if self.use_cuda:
            self.model.cuda()
        self.model.eval()
        # set model stepsize
        if 'r' in self.cp:
            self.model.decoder.set_r(self.cp['r'])
        # LOAD VOCODER MODEL
        self.vocoder_model = setup_generator(vocoder_config)
        self.vocoder_model.load_state_dict(torch.load(VOCODER_MODEL, map_location="cpu")["model"])
        self.vocoder_model.remove_weight_norm()
        self.vocoder_model.inference_padding = 0
        #ap_vocoder = AudioProcessor(**vocoder_config['audio'])    
        if use_cuda:
            self.vocoder_model.cuda()
        self.vocoder_model.eval()
        #get sample rate
        self.sample_rate = self.ap.sample_rate
        gc.collect(2)
    def tts(self,text,interactive=False,printable=False):
        figures=True
        t_1 = time.time()
        tmodel = copy.deepcopy(self.model)
        #tvoc = copy.deepcopy(self.vocoder_model)
        
        enable_chars = self.tts_config.enable_eos_bos_chars
        waveform, alignment, mel_spec, mel_postnet_spec, stop_tokens, inputs = synthesis(tmodel, text, self.tts_config, self.use_cuda, self.ap, self.speaker_id, style_wav=None, truncated=False, enable_eos_bos_chars=enable_chars)
        # mel_postnet_spec = ap._denormalize(mel_postnet_spec.T)
        del tmodel
        gc.collect(2)
        
        if not self.use_gl:
            waveform = self.vocoder_model.inference(torch.FloatTensor(mel_postnet_spec.T).unsqueeze(0))
            waveform = waveform.flatten()
        if self.use_cuda:
            waveform = waveform.cpu()
        else:
            waveform = waveform.numpy()
        #del tvoc
        
        if printable:
          rtf = (time.time() - t_1) / (len(waveform) / self.ap.sample_rate)
          tps = (time.time() - t_1) / len(waveform)
          usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
          print(waveform.shape)
          print(" > Run-time: {}".format(time.time() - t_1))
          print(" > Memory Used: {} MB".format(math.floor(usage/1024))) 
          print(" > Real-time factor: {}".format(rtf))
          print(" > Time per step: {}".format(tps))
        if interactive:
            IPython.display.display(IPython.display.Audio(waveform, rate=self.sample_rate)) 
        gc.collect(2)
        return alignment, mel_postnet_spec, stop_tokens, waveform
    def simpletts(self,text):
        _,_,_,wav = self.tts(text)
        return wav


# ## See it in action!

# In[8]:


def tryit(sample):
    # load the model
    ttsmodel = TTSModel("tts_model.pth.tar","config.json","vocoder_model.pth.tar","config_vocoder.json",False,False)
    # input sample and hear it!
    stuff = ttsmodel.tts(sample,True)
    del stuff
    del ttsmodel
#tryit("Bill got in the habit of asking himself “Is that thought true?” and if he wasn’t absolutely certain it was, he just let it go.")


# ## Process files and output to wav

# In[9]:


from pydub import AudioSegment
from functools import reduce
import re
from shutil import copyfile


# In[10]:


def preprocess(info):
    info = ' '.join(info.split('\n'))
    info = info.replace('- ','')
    into = '|'.join(map(lambda x: x, info.split('  ')))
    info = '|'.join(map(lambda x: x+'?', info.split('? ')))
    info = '|'.join(map(lambda x: x+'.', info.split('. ')))
    info = '|'.join(map(lambda x: x+'!', info.split('! ')))
    info = info.split('|')
    info = map(lambda x: ''.join(ch for ch in x if (ch.isalnum() or ch == ' ' or ch == '.' or ch == '?' or ch=='"' or ch=='\'' or ch=='”' or ch == '!')), info)
    info = [x for x in info if re.search('[a-zA-Z]', x)]
    #info = info[:-1]
    return list(filter(lambda x: len(x)>1,info))


# In[11]:


def writetofile(sample_rate, name, wav):
    scipy.io.wavfile.write(name,sample_rate,wav)
    return
def readfromfile(name):
    (_, wav) = scipy.io.wavfile.read(name)
    return wav


# In[12]:


def concat_sents(wav1,wav2):
    if not os.path.isfile('sil.wav'):
        AudioSegment.silence(duration=800).export('sil.wav',format='wav')
    sil = readfromfile('sil.wav')
    return np.concatenate((x,sil,y),axis=None)


# In[13]:


import concurrent.futures
import os.path

def speaksents(ttsmodel, sents, out, workers):
    def speak(sent):
        return ttsmodel.simpletts(sent)
    def nspeak(i):
        print(i)
        return ttsmodel.simpletts(sents[i])
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future = executor.map(speak, sents)
        #future = executor.map(nspeak, range(len(sents)))
        stuff = reduce(concat_sents,future)
        del future
        writetofile(ttsmodel.sample_rate, out+'.wav', stuff)
        del stuff
        gc.collect()
        return


# In[14]:


def wav2mp3(out):
    Audiosegment.from_wav(out+'.wav').export(out+'.mp3',format='mp3')


# In[15]:


def speaktofile(words,out,workers,ttsmodel):
    t_0 = time.time()
    initmodel = ttsmodel is None
    if initmodel:
        print('loading model')
        ttsmodel = TTSModel("tts_model.pth.tar","config.json","vocoder_model.pth.tar","config_vocoder.json",False,False)
    print('processing')
    sents = preprocess(words)
    print('reading '+str(len(sents))+' sentences')
    t_1 = time.time()
    speaksents(ttsmodel, sents, out, workers)
    print('reading took '+str((time.time()-t_1)/(60*60))+' h')
    if initmodel:
        del ttsmodel
    del sents
    gc.collect(2)
    print('converting to mp3')
    wav2mp3(out)
    print('done in '+str((time.time()-t_0)/(60*60))+' h')
    return


# In[16]:


def readtofile(filename,out,workers=2,ttsmodel=None):
    file = open(filename,"r")
    words = file.read()
    file.close()
    speaktofile(words,out,workers,ttsmodel)
    return


# ## Run it!

# In[17]:


#from google.colab import drive
#drive.mount('/content/gdrive')


# In[ ]:


readtofile("BeyondTheDoor.txt",'beyond')


# In[ ]:


#!cp "HowWeBecame.wav" "gdrive/My Drive/"

