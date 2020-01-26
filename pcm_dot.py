
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

    assert (np.all(f >= -8))
    assert (np.all(f <= 7))
    f = f + pow(2, (nbit - 1))

    gon = 1. / ron
    goff = 1./ roff

    pcm = [None] * nbit
    for bit in range(nbit):
        fb = np.bitwise_and(np.right_shift(f.astype(int), bit), 1)
        mean = (gon - goff) * fb + goff
        std = mean * var
        size = np.shape(f)
        assert(np.all(mean < 2 * gon))
        assert(np.all(mean > 0.5 * goff))
        pcm[bit] = np.random.normal(loc=mean, scale=std, size=size)

    return pcm
    
def adc(x, roff, ron, rows_per_read):
    gon = 1. / ron
    goff = 1./ roff
    
    m = (gon - goff)
    b = rows_per_read * goff
    y = np.clip(x - b, 0., np.inf) / m

    return np.round(y)

##############################

def conv(x, f, b, q, stride, pad1, pad2):
    Hi, Wi, Ci = np.shape(x)
    Fh, Fw, _, Co = np.shape(f)
    Ho = conv_output_length(Hi, Fh, 'same', stride)
    Wo = conv_output_length(Hi, Fw, 'same', stride)

    x = np.pad(array=x, pad_width=[[pad1,pad2], [pad1,pad2], [0,0]], mode='constant')

    p = np.zeros(shape=(Ho, Wo, Fh * Fw * Ci))
    y = np.zeros(shape=(Ho, Wo, Co))
    
    f_matrix = np.reshape(f, (Fh * Fw * Ci, Co))

    for h in range(Ho):        
        for w in range(Wo):
            patch = np.reshape(x[h*stride:(h*stride+Fh), w*stride:(w*stride+Fw), :], -1)
            assert(np.prod(np.shape(patch)) == np.prod(np.shape(f)[0:3])) # check {stride, pad1, pad2}
            p[h, w, :] = patch
            y[h, w, :] = conv_kernel(patch, f_matrix, b, q)

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
    
##############################
    
def pcm_conv(x, f, b, q, stride, pad1, pad2):
    Hi, Wi, Ci = np.shape(x)
    Fh, Fw, _, Co = np.shape(f)
    Ho = conv_output_length(Hi, Fh, 'same', stride)
    Wo = conv_output_length(Hi, Fw, 'same', stride)

    x = np.pad(array=x, pad_width=[[pad1,pad2], [pad1,pad2], [0,0]], mode='constant')

    p = np.zeros(shape=(Ho, Wo, Fh * Fw * Ci))
    y = np.zeros(shape=(Ho, Wo, Co))
    
    pcm = filter2pcm(f=f, nbit=4, var=0.06, roff=2000000, ron=25000)

    for h in range(Ho):        
        for w in range(Wo):
            patch = np.reshape(x[h*stride:(h*stride+Fh), w*stride:(w*stride+Fw), :], -1)
            assert(np.prod(np.shape(patch)) == np.prod(np.shape(f)[0:3])) # check {stride, pad1, pad2}
            p[h, w, :] = patch
            y[h, w, :] = pcm_conv_kernel(patch, pcm, b, q, 4, 4, 8)

    return y, p

def pcm_conv_kernel(patch, pcm, b, q, bit_per_act, bit_per_weight, rows_per_read):
    y = 0
    for row in range(0, len(patch), rows_per_read):        
        offset = 0
    
        r1 = row
        r2 = row + rows_per_read
        patch_rows = patch[r1:r2]
            
        for xb in range(bit_per_act):
            patch_rows_xb = np.bitwise_and(np.right_shift(patch_rows.astype(int), xb), 1)
            offset = offset + (np.sum(patch_rows_xb) << (xb + 3))
            
            for wb in range(bit_per_weight): # if all 16 rows are 1, then our adc is wrong.
                i = patch_rows_xb @ pcm[wb][r1:r2]
                d = adc(x=i, roff=2000000, ron=25000, rows_per_read=rows_per_read)
                y = y + np.left_shift(d.astype(int), xb + wb)

        y = y - offset

    assert(np.all(y < 2 ** 15))
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
3) problem was {offset/rows_per_read}... offset depends on this.
'''

x1 = np.copy(x_train[0])
f1 = np.copy(conv1)

x2 = np.copy(x_train[0])
f2 = np.copy(conv1)

y1, p1 = conv(x=x1, f=f1, b=conv1_bias, q=conv1_quant, stride=1, pad1=1, pad2=2)
y2, p2 = pcm_conv(x=x2, f=f2, b=conv1_bias, q=conv1_quant, stride=1, pad1=1, pad2=2)

print (np.count_nonzero(y1 == y2), np.prod(np.shape(y1)))
print (np.all(y1 == y2))

##############################
'''
# values = np.array([1])
# conv1 = np.random.choice(a=values, size=np.shape(conv1), replace=True).astype(int)

x1 = np.copy(x_train[0, 0:4, 0:4, :]).flatten()
f1 = np.copy(conv1)
f_matrix = np.reshape(f1, (4 * 4 * 3, 32))
y1 = conv_kernel(x1, f_matrix, 0, 0)

x2 = np.copy(x_train[0, 0:4, 0:4, :]).flatten()
f2 = np.copy(conv1)
pcm = filter2pcm(f=f2, nbit=4, var=0., roff=2000000, ron=25000)
y2 = pcm_conv_kernel(x2, pcm, 0, 0, 4, 4, 8)

print (y1)
print (y2)
'''
##############################





















