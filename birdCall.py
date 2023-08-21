import  torch.utils.data as data
import  os
import  os.path
import  errno
from tqdm import tqdm
import pandas as pd
import librosa
import numpy as np
from PIL import Image
from sklearn.utils import shuffle
import random

import pdb


class BirdCall(data.Dataset):

    RANDOM_SEED = 420
    SAMPLE_RATE = 32000
    SIGNAL_LENGTH = 30 # seconds
    SPEC_SHAPE = (28, 28) # height x width
    FMIN = 500
    FMAX = 12500
    MIN_RECORDINGS = 100 # threshold

    '''
    The items are (filename,category). The index of all the categories can be found in self.idx_classes
    Args:
    - root: the directory where the dataset will be stored
    - transform: how to transform the input
    - target_transform: how to transform the target
    - download: need to download the dataset
    '''

    def __init__(self):
        self.input_path = 'train_short_audio/'
        self.data_path = 'spectrogram_small_folder/'
        self.data_output_path = 'spectrogram_vals.npy'
        self.labels_output_path = 'spectrogram_labels.npy'
        self.csv_path = 'train_metadata.csv'

        self.most_represented_birds = []
        self.all_items = []
        self.name_to_idx = {}
        self.idx_to_name = {}
        self.bird_count = {}

        self.csv_input = pd.read_csv(self.csv_path,)
        self.csv_input = self.csv_input.query('rating>=4')

        self.find_most_represented_birds()
        
        if not os.path.isfile(self.data_output_path):
            if not os.path.exists(self.input_path):
                raise RuntimeError("Neither input data nor spectrograms are present.")
            else:
                self.process_recordings()

        self.fill_all_items(self.data_path)
        self.index_classes()
        x_vals = np.array(list(map(lambda x: x[0], self.all_items)))
        y_vals = np.array(list(map(lambda x: self.name_to_idx[x[1]], self.all_items)))
        np.save(self.data_output_path, x_vals)
        np.save(self.labels_output_path, y_vals)

    def index_classes(self):
        '''
        Populate dictionaries storing name to index mapping and vice versa
        '''
        assert len(self.all_items) > 0
        self.name_to_idx = {}
        self.idx_to_name = {}
        for idx, bird_name in enumerate(sorted(self.most_represented_birds)):
            self.name_to_idx[bird_name] = idx
            self.idx_to_name[idx] = bird_name

    def __len__(self):
        '''
        Returns total number of bird calls
        '''
        return len(self.all_items)

    def __get_item__(self, index):
        '''
        Returns image as nparray, index corresponding to bird species
        '''
        return self.all_items[index][0], self.idx_dict[self.all_items[index][1]]

    def get_species_from_spec_path(self, path):
        '''
        Return string that is the species name corresponding to a path
        '''
        return path.split('/')[1]

    def process_recordings(self):
        '''
        Output the spectrograms of recordings in self.input_path into self.data_path
        '''
        TRAIN = self.csv_input.query('primary_label in @self.most_represented_birds')
        with tqdm(total=len(TRAIN)) as pbar:
            for idx, row in TRAIN.iterrows():
                pbar.update(1)
                if row.primary_label in self.most_represented_birds:
                    audio_file_path = os.path.join(self.input_path, row.primary_label, row.filename)
                    self.create_spectrograms(audio_file_path, row.primary_label)

    def find_most_represented_birds(self):
        self.bird_count = {}
        for bird_species, count in zip(self.csv_input.primary_label.unique(), 
                                       self.csv_input.groupby('primary_label')['primary_label'].count().values):
            self.bird_count[bird_species] = count
        self.most_represented_birds = [key for key,value in self.bird_count.items() if value >= self.MIN_RECORDINGS]

    def fill_all_items(self, directory):
        '''
        Loop through all the files in a directory and return a list of [image file, species name]
        '''
        self.all_items = []
    
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                species_name = self.get_species_from_spec_path(file_path)
                self.all_items.append([np.array(Image.open(file_path)).astype(np.float32), species_name])

    def create_spectrograms(self, path, primary_label):
        sig, rate = librosa.load(path, sr=self.SAMPLE_RATE, offset=None)
    
        # randomly select 30 second interval: see https://github.com/ChristofHenkel/kaggle-birdclef2021-2nd-place/blob/26438069466242e9154aacb9818926dba7ddc7f0/data/ps_ds_8.py#L92
        call_length = np.size(sig) / self.SAMPLE_RATE
        max_offset = self.SIGNAL_LENGTH - call_length
        offset = np.random.randint(max(max_offset, 0) + 1)
        
        # repeat the audio as necessary
        min_length = self.SAMPLE_RATE * self.SIGNAL_LENGTH
    
        sig_regulated = np.copy(sig)
        while min_length > np.size(sig_regulated):
            sig_regulated = np.concatenate((sig_regulated, sig))
        sig_regulated = sig_regulated[:min_length]
        
        hop_length = int(self.SIGNAL_LENGTH * self.SAMPLE_RATE / (self.SPEC_SHAPE[1] - 1))
        mel_spec = librosa.feature.melspectrogram(y=sig_regulated, 
                                                  sr=self.SAMPLE_RATE, 
                                                  n_fft=1024, 
                                                  hop_length=hop_length, 
                                                  n_mels=self.SPEC_SHAPE[0], 
                                                  fmin=self.FMIN, 
                                                  fmax=self.FMAX)
    
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max) 
        
        # Normalize
        mel_spec -= mel_spec.min()
        mel_spec /= (mel_spec.max() + 1e-6)
        
        # Save as image file
        save_dir = os.path.join(self.data_path, primary_label)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, path.rsplit(os.sep, 1)[-1].rsplit('.', 1)[0] + '.png')
        im = Image.fromarray(mel_spec * 255.0).convert("L")
        im.save(save_path)
