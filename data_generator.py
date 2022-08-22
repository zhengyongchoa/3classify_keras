import numpy as np
import _pickle as cPickle
import keras
from keras.utils import np_utils
import config as cfg


def batch_data(paths):
    x_batch , y_batch =[] , []
    for file_name in paths:
        dir = file_name[0]
        x = cPickle.load(open(dir, 'rb'))
        y = int(dir.split('/')[-1].split('.')[0].split('_')[0][0])
        x_batch.append(x.astype(np.float32))
        # y_batch.append(y.astype(np.float32))
        y_batch.append(y-1) # 若num_classes=3，则y应为[0,1,2]

     # one-hot编码
    y_batch= np_utils.to_categorical(y_batch, cfg.num_classes)  #若num_classes=3，则y应为[0,1,2]


    return  np.array(x_batch) , np.array(y_batch)


class MY_Generator(keras.utils.Sequence):
    def __init__(self, filenames , batch_size):
        # image_filenames - 音频路径集合
        self.filenames = filenames
        self.batch_size = batch_size

    def __len__(self):
        # 计算 generator要生成的 batches 数
        return int(np.ceil(len(self.filenames) / self.batch_size) )

    def __getitem__(self, idx):
        # idx - 给定的 batch 数，以构建 batch 数据
        featpath = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        x_batch , y_batch =batch_data( featpath )
        return  x_batch , y_batch

    def get1(self, idx):
        # idx - 给定的 batch 数，以构建 batch 数据 [audio_batch, GT]
        featpath = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        x_batch , y_batch =batch_data( featpath )
        return  x_batch , y_batch



# class BalanceDataGenerator(object):
#     def __init__(self, batch_size, type, te_max_iter=100):
#         assert type in ['train', 'test']
#         self._batch_size_ = batch_size
#         self._type_ = type
#         self._te_max_iter_ = te_max_iter
#
#     def __len__(self):
#         return len()
#
#     def __getitem__(self, item):
#
#     def generate(self, xs):
#         batch_size = self._batch_size_
#         np.random.shuffle(xs)
#         fepath = xs[0:batch_size]
#
#         cnt = 0
#         while True:
#             batch_x = []
#             batch_y = []
#             for i1 in fepath:
#                 x , y = batch_feat(i1)
#                 batch_x.append(x)
#                 batch_y.append(y)
#
#                 cnt += 1
#                 if cnt == batch_size:
#                     cnt = 0
#                     yield batch_x, batch_y


# class RatioDataGenerator(object):
#     def __init__(self, batch_size, type, te_max_iter=100, verbose=1):
#         assert type in ['train', 'test']
#         self._batch_size_ = batch_size
#         self._type_ = type
#         self._te_max_iter_ = te_max_iter
#         self._verbose_ = verbose
#
#     def _get_lb_list(self, n_samples_list):
#         lb_list = []
#         for idx in range(len(n_samples_list)):
#             n_samples = n_samples_list[idx]
#             if n_samples < 1000:
#                 lb_list += [idx]
#             elif n_samples < 2000:
#                 lb_list += [idx] * 2
#             elif n_samples < 3000:
#                 lb_list += [idx] * 3
#             elif n_samples < 4000:
#                 lb_list += [idx] * 4
#             else:
#                 lb_list += [idx] * 5
#         return lb_list
#
#     def generate(self, xs, ys):   # 主体功能函数
#         batch_size = self._batch_size_
#         x = xs[0]
#         y = ys[0]
#         (n_samples, n_labs) = y.shape
#
#         n_samples_list = np.sum(y, axis=0)
#         lb_list = self._get_lb_list(n_samples_list)
#
#         if self._verbose_ == 1:
#             print("n_samples_list: %s" % (n_samples_list,))
#             print("lb_list: %s" % (lb_list,))
#             print("len(lb_list): %d" % len(lb_list))
#
#         index_list = []
#         for i1 in range(n_labs):
#             index_list.append(np.where(y[:, i1] == 1)[0])
#
#         for i1 in range(n_labs):
#             np.random.shuffle(index_list[i1])
#
#         queue = []
#         pointer_list = [0] * n_labs
#         len_list = [len(e) for e in index_list]
#         iter = 0
#         while True:
#             if (self._type_) == 'test' and (iter == self._te_max_iter_):
#                 break
#             iter += 1
#             batch_x = []
#             batch_y = []
#
#             while len(queue) < batch_size:
#                 random.shuffle(lb_list)
#                 queue += lb_list
#
#             batch_idx = queue[0 : batch_size]
#             queue[0 : batch_size] = []
#
#             n_per_class_list = [batch_idx.count(idx) for idx in range(n_labs)]
#
#             for i1 in range(n_labs):
#                 if pointer_list[i1] >= len_list[i1]:
#                     pointer_list[i1] = 0
#                     np.random.shuffle(index_list[i1])
#
#                 per_class_batch_idx = index_list[i1][pointer_list[i1] : min(pointer_list[i1] + n_per_class_list[i1], len_list[i1])]
#                 batch_x.append(x[per_class_batch_idx])
#                 batch_y.append(y[per_class_batch_idx])
#                 pointer_list[i1] += n_per_class_list[i1]
#             batch_x = np.concatenate(batch_x, axis=0)
#             batch_y = np.concatenate(batch_y, axis=0)
#             yield batch_x, batch_y

