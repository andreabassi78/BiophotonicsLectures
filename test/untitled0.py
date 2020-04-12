#from tensorflow.python.client import device_lib

#print(device_lib.list_local_devices())

# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]=""
#SET PATH=C:\ProgramData\NVIDIA GPU Computing Toolkit\v10.1\bin;%PATH%
#SET PATH=C:\ProgramData\NVIDIA GPU Computing Toolkit\v10.1\lib\x64;%PATH%

import tensorflow as tf

print (tf.__version__)

#print(tf.test.gpu_device_name())


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

print(tf.test.gpu_device_name())

#print(tf.config.list_physical_devices('GPU'))


# tf.test.is_gpu_available(
#     cuda_only=False,
#     min_cuda_compute_capability=None
# )