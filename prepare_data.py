from __future__ import print_function
import numpy as np
import sys
import soundfile
import os
import librosa
from scipy import signal
import pickle
import _pickle as cPickle
import time
import csv
import h5py
from sklearn import preprocessing
from sklearn import metrics
import config as cfg


# Read wav
def read_audio(path, target_fs=None):
    # (audio, fs)  = librosa.load( path )
    (audio, fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs
    
# Write wav
def write_audio(path, audio, sample_rate):
    soundfile.write(file=path, data=audio, samplerate=sample_rate)

# Create an empty folder
def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)

### Feature extraction. 
def extract_features(wav_dir, out_dir, recompute):
    """Extract log mel spectrogram features. 
    
    Args:
      wav_dir: string, directory of wavs. 
      out_dir: string, directory to write out features. 
      recompute: bool, if True recompute all features, if False skip existed
                 extracted features. 
                 
    Returns:
      None
    """
    fs = cfg.sample_rate
    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    
    create_folder(out_dir)
    names = [na for na in os.listdir(wav_dir) if na.endswith(".wav")]
    names = sorted(names)
    print("音频数据条数: %d" % len(names))
    label = wav_dir.split('/')[-1]

    # Mel filter bank
    melW = librosa.filters.mel(sr=fs,  n_fft=n_window, n_mels=64, fmin=0 , fmax=8000.)
    
    cnt = 0
    t1 = time.time()
    for na in names:
        wav_path = wav_dir + '/' + na
        out_path = out_dir + '/' + label + '_' + os.path.splitext(na)[0] + '.p'
        
        # Skip features already computed
        if recompute or (not os.path.isfile(out_path)):
            print(cnt, out_path)
            (wavdata, _) = read_audio(wav_path, fs)
            
            # Skip corrupted wavs
            if wavdata.shape[0] < 3*fs:
                print("File %s is corrupted!" % wav_path)
            else:

                audio = wavdata[fs : 3*fs]
                # 特征提取
                # 特征1： spectrogram
                ham_win = np.hamming(n_window)
                [f, t, x] = signal.spectral.spectrogram(
                                x=audio, 
                                window=ham_win,
                                nperseg=n_window, 
                                noverlap=n_overlap, 
                                detrend=False, 
                                return_onesided=True, 
                                mode='magnitude') 
                x = x.T
                x = np.dot(x, melW.T)
                x = np.log(x + 1e-8)
                x = x.astype(np.float32)

                # 提取特征2
                # f2 ,voiced_flag, voiced_probs= librosa.pyin( audio, sr = fs, frame_length = n_window ,hop_length =512, center=False, fmin = 65 ,fmax = 2000 )
                f2 ,voiced_flag, voiced_probs= librosa.pyin( audio, sr = fs, frame_length = n_window ,hop_length =512, fmin = 65 ,fmax = 2000 )
                # voiced_flag =np.array(  [int(elem) for elem in voiced_flag] )

                # 特区特征3
                f3 = librosa.feature.mfcc(audio, win_length =1024,  sr = fs, hop_length =512)

                # 特征优化：
                for i in range(len(voiced_flag ))  :
                     if  not voiced_flag[i] :
                          f2[i] = 0
                          f3[:,i] =0

                # 特征拼接20*63 +1*63
                f2 = f2[ np.newaxis, :]
                outfea =  np.concatenate( (f2 ,f3) )

                # Dump to pickle
                cPickle.dump(outfea, open(out_path, 'wb'), -1)
        cnt += 1
    print("抽取特征耗时！: %s" % (time.time() - t1,))

### Pack features of hdf5 file
def pack_features_to_hdf5(fe_dir, csv_path, out_path):
    """Pack extracted features to a single hdf5 file. 
    
    This hdf5 file can speed up loading the features. This hdf5 file has 
    structure:
       na_list: list of names
       x: bool array, (n_clips)
       y: float32 array, (n_clips, n_time, n_freq)
       
    Args: 
      fe_dir: string, directory of features. 
      csv_path: string | "", path of csv file. E.g. "testing_set.csv". If the 
          string is empty, then pack features with all labels False. 
      out_path: string, path to write out the created hdf5 file. 
      
    Returns:
      None
    """
    max_len = cfg.max_len
    create_folder(os.path.dirname(out_path))
    
    t1 = time.time()
    x_all, y_all, na_all = [], [], []
    
    if csv_path != "":    # Pack from csv file (training & testing from dev. data)
        with open(csv_path, 'rb') as f:
            reader = csv.reader(f)
            lis = list(reader)
        cnt = 0
        for li in lis:
            [na, bgn, fin, lbs, ids] = li
            if cnt % 100 == 0: print(cnt)
            na = os.path.splitext(na)[0]
            bare_na = 'Y' + na + '_' + bgn + '_' + fin # Correspond to the wav name. 
            fe_na = bare_na + ".p"
            fe_path = os.path.join(fe_dir, fe_na)
            
            if not os.path.isfile(fe_path):
                print("File %s is in the csv file but the feature is not extracted!" % fe_path)
            else:
                na_all.append(bare_na[1:] + ".wav") # Remove 'Y' in the begining. 
                x = cPickle.load(open(fe_path, 'rb'))
                x = pad_trunc_seq(x, max_len)
                x_all.append(x)
                ids = ids.split(',')
                y = ids_to_multinomial(ids)
                y_all.append(y)
            cnt += 1
    else:   # Pack from features without ground truth label (dev. data)
        names = os.listdir(fe_dir)
        names = sorted(names)
        for fe_na in names:
            bare_na = os.path.splitext(fe_na)[0]
            fe_path = os.path.join(fe_dir, fe_na)
            na_all.append(bare_na + ".wav")
            x = cPickle.load(open(fe_path, 'rb'))
            x = pad_trunc_seq(x, max_len)
            x_all.append(x)
            y_all.append(None)
        
    x_all = np.array(x_all, dtype=np.float32)
    y_all = np.array(y_all, dtype=np.bool)
    print("len(na_all): %d", len(na_all))
    print("x_all.shape: %s, %s" % (x_all.shape, x_all.dtype))
    print("y_all.shape: %s, %s" % (y_all.shape, y_all.dtype))
    
    with h5py.File(out_path, 'w') as hf:
        hf.create_dataset('na_list', data=na_all)
        hf.create_dataset('x', data=x_all)
        hf.create_dataset('y', data=y_all)
        
    print("Save hdf5 to %s" % out_path)
    print("Pack features time: %s" % (time.time() - t1,))
    
def ids_to_multinomial(ids):
    """Ids of wav to multinomial representation. 
    
    Args:
      ids: list of id, e.g. ['/m/0284vy3', '/m/02mfyn']
      
    Returns:
      1d array, multimonial representation, e.g. [1,0,1,0,0,...]
    """
    y = np.zeros(len(cfg.lbs))
    for id in ids:
        index = cfg.id_to_idx[id]
        y[index] = 1
    return y
    
def pad_trunc_seq(x, max_len):
    """Pad or truncate a sequence data to a fixed length. 
    
    Args:
      x: ndarray, input sequence data. 
      max_len: integer, length of sequence to be padded or truncated. 
      
    Returns:
      ndarray, Padded or truncated input sequence data. 
    """
    L = len(x)
    shape = x.shape
    if L < max_len:
        pad_shape = (max_len - L,) + shape[1:]
        pad = np.zeros(pad_shape)
        x_new = np.concatenate((x, pad), axis=0)
    else:
        x_new = x[0:max_len]
    return x_new
    
### Load data & scale data
def load_hdf5_data(hdf5_path, verbose=1):
    """Load hdf5 data. 
    
    Args:
      hdf5_path: string, path of hdf5 file. 
      verbose: integar, print flag. 
      
    Returns:
      x: ndarray (np.float32), shape: (n_clips, n_time, n_freq)
      y: ndarray (np.bool), shape: (n_clips, n_classes)
      na_list: list, containing wav names. 
    """
    t1 = time.time()
    with h5py.File(hdf5_path, 'r') as hf:
        x = np.array(hf.get('x'))
        y = np.array(hf.get('y'))
        na_list = list(hf.get('na_list'))
        
    if verbose == 1:
        print("--- %s ---" % hdf5_path)
        print("x.shape: %s %s" % (x.shape, x.dtype))
        print("y.shape: %s %s" % (y.shape, y.dtype))
        print("len(na_list): %d" % len(na_list))
        print("Loading time: %s" % (time.time() - t1,))
        
    return x, y, na_list

def calculate_scaler(hdf5_path, out_path):
    """Calculate scaler of input data on each frequency bin. 
    
    Args:
      hdf5_path: string, path of packed hdf5 features file. 
      out_path: string, path to write out the calculated scaler. 
      
    Returns:
      None. 
    """
    t1 = time.time()
    (x, y, na_list) = load_hdf5_data(hdf5_path, verbose=1)
    (n_clips, n_time, n_freq) = x.shape
    x2d = x.reshape((n_clips * n_time, n_freq))
    scaler = preprocessing.StandardScaler().fit(x2d)
    print("Mean: %s" % (scaler.mean_,))
    print("Std: %s" % (scaler.scale_,))
    print("Calculating scaler time: %s" % (time.time() - t1,))
    pickle.dump(scaler, open(out_path, 'wb'))
    
def do_scale(x3d, scaler_path, verbose=1):
    """Do scale on the input sequence data. 
    
    Args:
      x3d: ndarray, input sequence data, shape: (n_clips, n_time, n_freq)
      scaler_path: string, path of pre-calculated scaler. 
      verbose: integar, print flag. 
      
    Returns:
      Scaled input sequence data. 
    """
    t1 = time.time()
    scaler = pickle.load(open(scaler_path, 'rb'))
    (n_clips, n_time, n_freq) = x3d.shape
    x2d = x3d.reshape((n_clips * n_time, n_freq))
    x2d_scaled = scaler.transform(x2d)
    x3d_scaled = x2d_scaled.reshape((n_clips, n_time, n_freq))
    if verbose == 1:
        print("Scaling time: %s" % (time.time() - t1,))
    return x3d_scaled

def make_csv( dir , modetype ) :
    fe_dir = os.path.join(dir , modetype)

    f_all = []

    names = os.listdir(fe_dir)
    names = sorted(names)
    for fe_na in names:
        fe_path = os.path.join(fe_dir, fe_na)
        fe_path = [fe_path]
        f_all.append(fe_path)

    out = modetype + '.csv'
    out_path = os.path.join( dir , out )
    with open( out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in f_all :
            writer.writerow(row)

    print("Save csv to %s" % out_path)


def load_csv(csv_path):
    filpath =[]
    if csv_path != "":  # Pack from csv file (training & testing from dev. data)
        # with open(csv_path, 'rb') as f:
        #     reader = csv.reader(f)
        #     lis = list(reader)
        f = open(csv_path)
        data = csv.reader(f)  # ①
        for line in data:
            tmp = line
            filpath.append(tmp)

    else:
        print('没有特征文件！')

    return filpath

### Main function
if __name__ == '__main__':


    wavpath = ['train', 'val', 'test']
    # for i in wavpath:
    #     mode = i
    #     feat_dir = os.path.join(cfg.feat_dir, mode)
    #     create_folder(feat_dir)
    #     for type in  range(1,4):
    #         type =str(type)
    #         print( type )
    #         wav_dir = os.path.join(cfg.wav_dir, mode, type)
    #         extract_features(wav_dir, feat_dir, 1)

    for i in wavpath:
        name = i
        make_csv(cfg.feat_dir, name)
