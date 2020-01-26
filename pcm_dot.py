
import numpy as np
import tensorflow as tf

from conv_utils import conv_output_length

##############################

def quantize_activations(a):
  scale = (np.max(a) - np.min(a)) / (15 - 0)
  a = a / scale
  a = np.floor(a)
  a = np.clip(a, 0, 15)
  return a, scale
  
##############################

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

train_examples = 50000
test_examples = 10000

assert(np.shape(x_train) == (50000, 32, 32, 3))
# x_train = x_train - np.mean(x_train, axis=0, keepdims=True)
# x_train = x_train / np.std(x_train, axis=0, keepdims=True)
x_train, _ = quantize_activations(x_train)
y_train = tf.keras.utils.to_categorical(y_train, 10)

assert(np.shape(x_test) == (10000, 32, 32, 3))
# x_test = x_test - np.mean(x_test, axis=0, keepdims=True)
# x_test = x_test / np.std(x_test, axis=0, keepdims=True)
x_test, _ = quantize_activations(x_test)
y_test = tf.keras.utils.to_categorical(y_test, 10)

##############################

weights = np.load('cifar10_conv.npy', allow_pickle=True).item()

conv1 = weights['conv1']
conv1_bias = np.zeros_like(weights['conv1_bias'])
conv1_quant = weights['conv1_scale']

##############################

def filter2pcm(f, nbit, var, roff, ron):
    Fh, Fw, Ci, Co = np.shape(f)
    f = np.reshape(f, (Fh * Fw * Ci, Co))

    gon = 1. / ron
    goff = 1./ roff

    pcm = [None] * nbit
    for b in range(nbit):
        fb = np.bitwise_and(np.right_shift(f.astype(int), b), 1)
        mean = fb * (gon - goff) + goff
        std = mean * var
        size = np.shape(f)
        pcm[b] = np.random.normal(loc=mean, scale=std, size=size)

    return pcm
    
def adc(x, roff, ron):
    gon = 1. / ron
    goff = 1./ roff
    
    y = (x - (8 * goff)) / (gon - goff)
    y = np.round(y)
    return y

##############################

def conv(x, f, b, q, stride, pad1, pad2, kernel='digital'):
    Hi, Wi, Ci = np.shape(x)
    Fh, Fw, _, Co = np.shape(f)
    Ho = conv_output_length(Hi, Fh, 'same', stride)
    Wo = conv_output_length(Hi, Fw, 'same', stride)

    x = np.pad(array=x, pad_width=[[pad1,pad2], [pad1,pad2], [0,0]], mode='constant')

    p = np.zeros(shape=(Ho, Wo, Fh * Fw * Ci))
    y = np.zeros(shape=(Ho, Wo, Co))

    for h in range(Ho):        
        for w in range(Wo):
            patch = np.reshape(x[h*stride:(h*stride+Fh), w*stride:(w*stride+Fw), :], -1)
            assert(np.prod(np.shape(patch)) == np.prod(np.shape(f)[0:3])) # check {stride, pad1, pad2}
            p[h, w, :] = patch
            if kernel == 'digital': 
                f_matrix = np.reshape(f, (Fh * Fw * Ci, Co))
                y[h, w, :] = conv_kernel(patch, f_matrix, b, q)
            elif kernel == 'pcm':
                pcm = filter2pcm(f, 4, 0.035, 25000, 2000000)
                y[h, w, :] = pcm_conv_kernel(patch, pcm, b, q, 4, 4, 8)
            else: 
                assert (False)

    return y, p

def conv_kernel(patch, f, b, q):
    y = patch @ f
    assert(np.all(np.absolute(y) < 2 ** 15))
    y = y + b
    y = y * (y > 0)
    y = y // q 
    y = np.clip(y, 0, 15)
    y = y.astype(int)
    return y

def pcm_conv_kernel(patch, pcm, b, q, bit_per_act, bit_per_weight, rows_per_read):
    y = 0
    offset = 0

    for xb in range(bit_per_act):
        patch_xb = np.bitwise_and(np.right_shift(patch.astype(int), xb), 1)
        offset = offset + (np.sum(patch_xb) << (xb + 3))
        for wb in range(bit_per_weight): # if all 16 rows are 1, then our adc is wrong.
            for row in range(0, rows_per_read, len(patch_xb)):
                r1 = row * rows_per_read
                r2 = r1 + rows_per_read
                i = patch_xb[r1:r2] @ pcm[wb][r1:r2]
                d = adc(i, 25000, 2000000)
                y = y + np.left_shift(d.astype(int), xb + wb)

    assert(np.all(y < 2 ** 15))
    y = y - offset
    y = y + b
    y = y * (y > 0)
    y = y // q 
    y = np.clip(y, 0, 15)
    y = y.astype(int)
    return y

##############################

'''
todo items:
1) if all 16 rows are 1, then our adc is wrong.
2) make sure shallow copies not breaking things with {x, filter}

'''

y1, p1 = conv(x=x_train[0], f=conv1, b=conv1_bias, q=conv1_quant, stride=1, pad1=1, pad2=2, kernel='digital')
y2, p2 = conv(x=x_train[0], f=conv1, b=conv1_bias, q=conv1_quant, stride=1, pad1=1, pad2=2, kernel='pcm')

print (np.shape(y1))
print (np.shape(y2))

print (np.all(y1 == y2))

##############################





