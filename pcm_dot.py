
import numpy as np
import tensorflow as tf

from quant import quantize_conv_activations

from conv_utils import conv_output_length

##############################

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

train_examples = 50000
test_examples = 10000

assert(np.shape(x_train) == (50000, 32, 32, 3))
# x_train = x_train - np.mean(x_train, axis=0, keepdims=True)
# x_train = x_train / np.std(x_train, axis=0, keepdims=True)
x_train, _ = quantize_conv_activations(x_train)
y_train = tf.keras.utils.to_categorical(y_train, 10)

assert(np.shape(x_test) == (10000, 32, 32, 3))
# x_test = x_test - np.mean(x_test, axis=0, keepdims=True)
# x_test = x_test / np.std(x_test, axis=0, keepdims=True)
x_test, _ = quantize_conv_activations(x_test)
y_test = tf.keras.utils.to_categorical(y_test, 10)

##############################

weights = np.load('cifar10_conv.npy', allow_pickle=True).item()

conv1 = weights['conv1']
conv1_bias = np.zeros_like(weights['conv1_bias'])
conv1_quant = weights['conv1_scale']

def filter2pcm(f, nbit, var, roff, ron):
    Fh, Fw, Ci, Co = np.shape(f)
    f = np.reshape(f, (Fh * Fw * Ci, Co))

    pcm = [None] * nbit
    for b in range(nbit):
        fb = np.bitwise_and(np.right_shift(f.astype(int), b), 1)
        mean = fb * (roff - ron) + ron
        std = mean * var
        size = np.shape(f)
        pcm[b] = np.random.normal(loc=mean, scale=std, size=size)

    return pcm

##############################

# pcm = filter2pcm(conv1, 4, 0.035, 200000, 25000)

##############################

def conv(x, f, b, q, stride, pad1, pad2):
    Hi, Wi, Ci = np.shape(x)
    Fh, Fw, _, Co = np.shape(f)
    Ho = conv_output_length(Hi, Fh, 'same', stride)
    Wo = conv_output_length(Hi, Fw, 'same', stride)

    print (np.shape(x), pad1, pad2)

    x = np.pad(array=x, pad_width=[[pad1,pad2], [pad1,pad2], [0,0]], mode='constant')

    p = np.zeros(shape=(Ho, Wo, Fh * Fw * Ci))
    y = np.zeros(shape=(Ho, Wo, Co))
    
    f_matrix = np.reshape(f, (Fh * Fw * Ci, Co))

    for h in range(Ho):        
        for w in range(Wo):
            patch = np.reshape(x[h*stride:(h*stride+Fh), w*stride:(w*stride+Fw), :], -1)
            assert(np.prod(np.shape(patch)) == np.shape(f_matrix)[0])
            p[h, w, :] = patch
            y[h, w, :] = bin_conv_kernel(patch, f_matrix, b, q)

    return y, p

'''
def conv_kernel(patch, f, b, q):
    y = patch @ f
    assert(np.all(np.absolute(y) < 2 ** 15))
    y = y + b
    y = y * (y > 0)
    y = y.astype(int)
    # y = np.bitwise_and(y, 15)
    # quant cannot be 1
    y = y // q 
    y = np.clip(y, 0, 15)
    return y
'''
##############################
'''
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
'''

def conv_kernel(patch, f, b, q, bit_per_val, bit_per_weight):
    y = 0
    offset = 0

    for xb in range(bit_per_val):
        patch_xb = np.bitwise_and(np.right_shift(patch.astype(int), xb), 1)
        offset = offset + (np.sum(patch_xb) << (xb + 3))

        # if all 16 rows are 1, then our adc is wrong.
        for wb in range(bit_per_weight):
            pcm_wb = np.bitwise_and(np.right_shift(pcm.astype(int), wb), 1)

            dot = patch_xb @ pcm_wb
            y = y + np.left_shift(dot.astype(int), xb + wb)

    assert(np.all(y < 2 ** 15))
    y = y - offset
    y = y + b
    y = y * (y > 0)
    y = y // q 
    y = y.astype(int)
    return y

##############################

y = conv(x=x_train[0], f=conv1, b=conv1_bias, q=conv1_quant, stride=1, pad1=1, pad2=1)










