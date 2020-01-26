
import numpy as np
np.set_printoptions(threshold=np.inf)
import tensorflow as tf

from conv_utils import conv_output_length
from init_tensor import init_filters
from batchnorm import batchnorm

###########################################

def debug(mat):
    mat = np.reshape(mat, -1)
    assert(np.all(mat == mat[0]))
    print (mat[0], end=' ')

###########################################

def conv(x, pcm, bit_per_val, bit_per_weight):
    Hi, Wi, Ci = np.shape(x)
    _, Co = np.shape(pcm)
    Ho = conv_output_length(Hi, 4, 'valid', 1)
    Wo = conv_output_length(Hi, 4, 'valid', 1)

    p = np.zeros(shape=(Ho, Wo, 4 * 4 * Ci))
    y = np.zeros(shape=(Ho, Wo, Co))

    for h in range(Ho):        
        for w in range(Wo):
            patch = np.reshape(x[h:(h+4), w:(w+4), :], -1)
            assert(np.prod(np.shape(patch)) == np.shape(pcm)[0])
            p[h, w, :] = patch
            y[h, w, :] = bin_conv_kernel(patch, pcm, bit_per_val, bit_per_weight)

    return y, p

###########################################

# want to have this and bin kernel to make sure we doing bin kernel right. 
def conv_kernel(patch, pcm, bit_per_val, bit_per_weight):
    return None

###########################################

def bin_conv_kernel(patch, pcm, bit_per_val, bit_per_weight):
    y = 0
    offset = 0

    for xb in range(bit_per_val):
        patch_xb = np.bitwise_and(np.right_shift(patch.astype(int), xb), 1)
        offset = offset + (np.sum(patch_xb) << (xb + 3))

        for wb in range(bit_per_weight):
            pcm_wb = np.bitwise_and(np.right_shift(pcm.astype(int), wb), 1)

            # there will be big issues here.
            # if all 16 rows are 1, then our adc is wrong.
            # this dosnt seem to happen often
            # in sim, we quantize 16 -> 15 ... in emu we do not.
            
            # in order to do this in emu, we have to loop over the 16 rows at a time and quantize them
            
            dot = patch_xb @ pcm_wb
            y = y + np.left_shift(dot.astype(int), xb + wb)

    # its great to do offset at the end, but can easily get an overflow here.
    assert(np.all(y < 2 ** 15))
    y = y - offset
    y = y * (y > 0)
    y = y.astype(int)
    # y = np.bitwise_and(y, 15)

    return y
    
###########################################

def dot(patch, pcm, bit_per_val, bit_per_weight):
    y = 0
    offset = 0

    for xb in range(bit_per_val):
        patch_xb = np.bitwise_and(np.right_shift(patch.astype(int), xb), 1)
        offset = offset + (np.sum(patch_xb) << (xb + 3))

        for wb in range(bit_per_weight):
            pcm_wb = np.bitwise_and(np.right_shift(pcm.astype(int), wb), 1)

            # there will be big issues here.
            # if all 16 rows are 1, then our adc is wrong.
            # this dosnt seem to happen often
            # in sim, we quantize 16 -> 15 ... in emu we do not.
            dot = patch_xb @ pcm_wb
            y = y + np.left_shift(dot.astype(int), xb + wb)

    # its great to do offset at the end, but can easily get an overflow here.
    assert(np.all(y < 2 ** 15))
    y = y - offset
    y = y * (y > 0)
    y = y.astype(int)
    # y = np.bitwise_and(y, 15)

    return y

###########################################

num_examples = 3
num_layers = 2

row_per_read = 16
bit_per_val = 4
bit_per_weight = 4

filter_matrix_shape = [
[4*4*3, 32],
[4*4*32,32]
]

filter_shape = [
[4,4, 3,32],
[4,4,32,32]
]

patch_shape = [
[7*7, 4*4*3 ],
[4*4, 4*4*32]
]

y_shape = [
[7*7, 32],
[4*4, 32]
]

###########################################

rows = 16
vdd = 1.0

ron = 1e3
roff = 1e5
gon = 1. / ron
goff = 1. / roff

###########################################

pcm = [None] * num_layers

for ii in range(num_layers):
    # states = np.array(range(2 ** bit_per_weight))
    states = np.array(range(1, 2 ** bit_per_weight))
    # beware: using something like this causes the 16 issue!!!
    # states = np.array([8, 9])
    pcm[ii] = np.random.choice(a=states, size=filter_matrix_shape[ii], replace=True)

###########################################

(x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()
assert(np.shape(x_train) == (50000, 32, 32, 3))
x_train = x_train[:, 0:10, 0:10, :]

x_train = np.floor(x_train / np.max(x_train, axis=(0,1,2), keepdims=True) * 15)
x_train = x_train.astype(int)
# beware: using something like this causes the 16 issue!!!
# x_train = np.ones_like(x_train)

###########################################

patch = [[None for col in range(num_layers)] for row in range(num_examples)] 
yout  = [[None for col in range(num_layers)] for row in range(num_examples)] 

for ii in range(num_examples):
    for jj in range(num_layers):
        # xin = x_train[ii] if (jj == 0) else yout[ii][jj-1]
        xin = x_train[ii] if (jj == 0) else np.bitwise_and(yout[ii][jj-1].astype(int), 15)
        yout[ii][jj], patch[ii][jj] = conv(xin, pcm[jj], bit_per_val, bit_per_weight)

###########################################

for ii in range(num_examples):
    np.savetxt("x%d.csv" % (ii+1), np.reshape(x_train[ii], -1), fmt='%d', delimiter=" ")

for ii in range(num_examples):
    for jj in range(num_layers):
        np.savetxt("patch%d_%d.csv" % (ii+1, jj+1), np.reshape(patch[ii][jj], patch_shape[jj]), fmt='%d', delimiter=" ")
        np.savetxt("yout%d_%d.csv" % (ii+1, jj+1), np.reshape(yout[ii][jj], y_shape[jj]), fmt='%d', delimiter=" ")

###########################################

# start dense stuff here I think.

dense_shape = [512, 256]

# states = np.array(range(2 ** bit_per_weight))
states = np.array(range(1, 2 ** bit_per_weight))
dense_pcm = np.random.choice(a=states, size=dense_shape, replace=True).astype(int)

# if we ever do multiple matmul layers remember to do:
# [[None for col in range(num_layers)] for row in range(num_examples)] 
dense_xin = [None] * num_examples 
dense_yout = [None] * num_examples 

for ii in range(num_examples):
    # dense_xin[ii] = np.reshape(yout[ii][-1], -1)
    dense_xin[ii] = np.reshape(np.bitwise_and(yout[ii][-1].astype(int), 15), -1)
    dense_yout[ii] = dot(dense_xin[ii], dense_pcm, bit_per_val, bit_per_weight)

###########################################

for ii in range(num_examples):
    np.savetxt("yout%d_3.csv" % (ii+1), np.reshape(dense_yout[ii], -1), fmt='%d', delimiter=" ")

###########################################

weights = []

###########################################

# we swap col and row in filter bc out of pixel fifo we deal col as base unit, not row.

weight = np.reshape(pcm[0], filter_shape[0])
weight = np.transpose(weight, (2,0,1,3))
weight = np.reshape(weight, (3, 4*4*1, 32))

zeros = np.zeros(shape=(5, 4*4*1, 32)).astype(int)
weight = np.concatenate((weight, zeros), axis=0)

zeros = np.zeros(shape=(8, 4*4*3, 32)).astype(int)
weight = np.concatenate((weight, zeros), axis=1)

weights.append(weight)

###########################################

'''
conv_weight = np.transpose(conv_weight, (2,1,0,3))
conv_weight = np.reshape(conv_weight, (8,4,4,4,32))
conv_weight = np.transpose(conv_weight, (0,2,3,1,4))
'''

for ii in range(1, num_layers):
    weight = pcm[ii]
    weight = np.reshape(weight, filter_shape[ii])
    weight = np.transpose(weight, (2,1,0,3))
    #scramble
    weight = np.reshape(weight, (4,8,4,4,32))
    weight = np.transpose(weight, (1,2,3,0,4))
    weight = np.reshape(weight, (8, 4*4*4, 32))
    #end scramble
    weights.append(weight)

weights = np.concatenate(weights, axis=1)

print (np.shape(weights))

###########################################

dense_weight = dense_pcm

dense_weight = np.reshape(dense_weight, (4, 4, 32, 256))
dense_weight = np.transpose(dense_weight, (1, 0, 2, 3))
dense_weight = np.reshape(dense_weight, (512, 256))

dense_weight = np.reshape(dense_weight, (512, 8, 32))
dense_weight = np.transpose(dense_weight, (1, 0, 2))

print (np.shape(dense_weight))

weights = np.concatenate((weights, dense_weight), axis=1)

###########################################

for ii in range(len(weights)):
    np.savetxt("pcm%d.csv" % (ii+1), weights[ii], fmt='%d', delimiter=" ")

###########################################













































