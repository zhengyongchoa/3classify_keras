import csv ,os ,sys
import numpy as np
import pandas as pd
import test
import _pickle as cPickle

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)


import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.test.is_gpu_available()



out_path = os.path.join( 'feature ' , 'test1.csv')
f_all = [['豆瓣排名'], ['电影名称'],['类别'],['评分']]

# with open( out_path, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     for row in f_all :
#         writer.writerow(row)

# df = pd.read_csv('feature /train.csv' , 'header=None')

# with open('feature /train.csv', 'rb') as f:
#     reader = csv.reader(f)
#     lis = list(reader)

# f = open('feature /train.csv' )
# data = csv.reader(f)    # ①
# for line in data:
#     print(line)


# dir = 'feature /train/3record441952 (12th copy).p'
# x = cPickle.load(open(dir, 'rb'))
# y = dir.split('/')[-1].split('.')[0].split('_')[0]
#
# print(x)




















# def pack_features_to_hdf5(fe_dir, csv_path, out_path):
#     """Pack extracted features to a single hdf5 file.
#
#     This hdf5 file can speed up loading the features. This hdf5 file has
#     structure:
#        na_list: list of names
#        x: bool array, (n_clips)
#        y: float32 array, (n_clips, n_time, n_freq)
#
#     Args:
#       fe_dir: string, directory of features.
#       csv_path: string | "", path of csv file. E.g. "testing_set.csv". If the
#           string is empty, then pack features with all labels False.
#       out_path: string, path to write out the created hdf5 file.
#
#     Returns:
#       None
#     """
#     max_len = cfg.max_len
#     create_folder(os.path.dirname(out_path))
#
#     t1 = time.time()
#     x_all, y_all, na_all = [], [], []
#
#     if csv_path != "":  # Pack from csv file (training & testing from dev. data)
#         with open(csv_path, 'rb') as f:
#             reader = csv.reader(f)
#             lis = list(reader)
#         cnt = 0
#         for li in lis:
#             [na, bgn, fin, lbs, ids] = li
#             if cnt % 100 == 0: print(cnt)
#             na = os.path.splitext(na)[0]
#             bare_na = 'Y' + na + '_' + bgn + '_' + fin  # Correspond to the wav name.
#             fe_na = bare_na + ".p"
#             fe_path = os.path.join(fe_dir, fe_na)
#
#             if not os.path.isfile(fe_path):
#                 print("File %s is in the csv file but the feature is not extracted!" % fe_path)
#             else:
#                 na_all.append(bare_na[1:] + ".wav")  # Remove 'Y' in the begining.
#                 x = cPickle.load(open(fe_path, 'rb'))
#                 x = pad_trunc_seq(x, max_len)
#                 x_all.append(x)
#                 ids = ids.split(',')
#                 y = ids_to_multinomial(ids)
#                 y_all.append(y)
#             cnt += 1
#     else:  # Pack from features without ground truth label (dev. data)
#         names = os.listdir(fe_dir)
#         names = sorted(names)
#         for fe_na in names:
#             bare_na = os.path.splitext(fe_na)[0]
#             fe_path = os.path.join(fe_dir, fe_na)
#             na_all.append(bare_na + ".wav")
#             x = cPickle.load(open(fe_path, 'rb'))
#             x = pad_trunc_seq(x, max_len)
#             x_all.append(x)
#             y_all.append(None)
#
#     x_all = np.array(x_all, dtype=np.float32)
#     y_all = np.array(y_all, dtype=np.bool)
#     print("len(na_all): %d", len(na_all))
#     print("x_all.shape: %s, %s" % (x_all.shape, x_all.dtype))
#     print("y_all.shape: %s, %s" % (y_all.shape, y_all.dtype))
#
#     with h5py.File(out_path, 'w') as hf:
#         hf.create_dataset('na_list', data=na_all)
#         hf.create_dataset('x', data=x_all)
#         hf.create_dataset('y', data=y_all)
#
#     print("Save hdf5 to %s" % out_path)
#     print("Pack features time: %s" % (time.time() - t1,))