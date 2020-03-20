import sys
# import numpy as np
import tensorflow as tf
print(tf.__version__,'<<<<<<<<<<<<<<<< tensorflow version')
from datetime import datetime
device_name = sys.argv[1]  # Choose device from cmd line. Options: gpu or cpu
shape = (int(sys.argv[2]), int(sys.argv[2]))
if device_name == "gpu":
    device_name = "/gpu:0"
else:
    device_name = "/cpu:0"


startTime = datetime.now()
session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
with tf.device(device_name):
    random_matrix = tf.random.uniform(shape=shape, minval=0, maxval=1)
    dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
    sum_operation = tf.reduce_sum(dot_operation)

result = sum_operation
print(result)
    # It can be hard to see the results on the terminal with lots of output -- add some newlines to improve readability.
print("\n" * 5)
print("Shape:", shape, "Device:", device_name)
print("Time taken:", str(datetime.now() - startTime))
session.close()