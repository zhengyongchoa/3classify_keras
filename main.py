from __future__ import print_function
import sys
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory)

import tensorflow.compat.v1 as tf  # 使用1.0版本的方法
tf.disable_v2_behavior()  # 禁用2.0版本的方法

from train import train

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
#
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if __name__ == '__main__':

    train()