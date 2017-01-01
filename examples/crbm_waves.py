"""
A simple example of how to train a predictive time-series model with CRBM

Created by Omid Alemi
"""

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import PIL.Image as Image

import numpy as np
import tensorflow as tf

from xrbm.models.crbm import CRBM
from xrbm.utils.vizutils import tile_raster_images

 # Set up the parameters
snapshot_dir = './logs/'
snapshot_freq = 0
num_vis = 4
num_hid = 20
timesteps = 100
batch_size=50
cd_k=10
wdecay=0.0002
activation=tf.nn.sigmoid
vis_type='gaussian'


# Create some toy sequences with sin waves

FREQS = [0.1, 0.5, 0.7, 1.2]
AMPS = [2, 1, 0.5, 1.5]
NSAMPLE = 100
SEQ_LEN = 600

time_data = np.arange(SEQ_LEN) / 10

X_data = []

print('Making dummy time series...')
for i in range(NSAMPLE):
    r_data = np.float32(np.random.rand(SEQ_LEN, num_vis)) / 20 # random noise
    x = np.asarray([np.float32(np.sin(FREQS[d] * time_data + np.random.rand()) * (AMPS[d]+np.random.rand()/20)) for d in range(num_vis)]).T
    X_data.append(x)

X_data = np.asarray(X_data)

X_data_flat = np.concatenate([m for m in X_data], axis=0)

data_mean = np.mean(X_data_flat, axis=0)
data_std = np.std(X_data_flat, axis=0)

X_data_normalized = data_norm = [(d - data_mean) / data_std for d in X_data]

def makeSeqHistory(seqs, order, step=1):
    history = []
    outputs = []
    for m in seqs:
        for i in range(0, len(m)-order, step):
            history.append(m[i:i+order].flatten())
            outputs.append(m[i+order])

    return np.asarray(history), np.asarray(outputs)


cond_data, visible_data = makeSeqHistory(X_data_normalized, timesteps)

num_cond = cond_data[0].shape[0]

c1 = CRBM(num_vis=num_vis, num_cond=num_cond , num_hid=num_hid, vis_type=vis_type, name='crbm_toy', activation=activation)

gen_cond = tf.placeholder(tf.float32, shape=[1, cond_data.shape[1]], name='gen_cond_data')
gen_init = tf.placeholder(tf.float32, shape=[1, visible_data.shape[1]], name='gen_init_data')
gen_op = c1.make_prediction(gen_cond, gen_init, 60)


# Initialize the tf variables
init = tf.global_variables_initializer()

# Lunch the session
sess = tf.Session()
sess.run(init)

## Train
learning_rate = 0.001
momentum = 0

print('-'*80)
print('Training %s CRBM with %i %s units'%(vis_type, num_hid, activation))
print('lr: %1.3f, batchsize: %i, cd: %i, wdecay: %f, mom: %1.1f'%(learning_rate, batch_size, cd_k, wdecay, momentum))
# print('sparsity target: %1.4f, sparsity cost: %1.3f, sparse_decay %1.3f'%(sparse_target, sparse_cost, sparse_decay))
print('-'*80)
# print('Dataset: %s'%(mocdataset))
print('Model order: %i'%(timesteps))
print('Training data size: %s sequences of length %i'%(len(visible_data), timesteps))
print('-'*80)

## for first 6 epoches, don't use momentum
W, A, B, vb, hb = c1.train(sess,
         input_data=visible_data,
         cond_data=cond_data,
         training_epochs=15,
         learning_rate=learning_rate,
         batch_size=batch_size,
         cd_k=cd_k,
         wdecay=wdecay,
         momentum=momentum)

# now add the momentum for the rest of the training

learning_rate = 0.001
momentum = 0.9

print('-'*80)
print('Training %s CRBM with %i %s units'%(vis_type, num_hid, activation))
print('lr: %1.3f, batchsize: %i, cd: %i, wdecay: %f, mom: %1.1f'%(learning_rate, batch_size, cd_k, wdecay, momentum))
# print('sparsity target: %1.4f, sparsity cost: %1.3f, sparse_decay %1.3f'%(sparse_target, sparse_cost, sparse_decay))
print('-'*80)
# print('Dataset: %s'%(mocdataset))
print('Model order: %i'%(timesteps))
print('Training data size: %s sequences of length %i'%(len(visible_data), timesteps))
print('-'*80)

W, A, B, vb, hb = c1.train(sess,
         input_data=visible_data,
         cond_data=cond_data,
         training_epochs=400,
         learning_rate=0.001,
         batch_size=batch_size,
         cd_k=cd_k,
         wdecay=wdecay,
         momentum=momentum)


######
# now let's generate somthing
gen_sample = []
gen_hidden = []
initcond = []

gen_init_frame = 100
num_gen = 500

for f in range(timesteps):
    gen_sample.append(np.reshape(visible_data[gen_init_frame+f], [1, num_vis]))

# gen_cond = deque(test_cond[gen_init_frame])
# gen_init = test_input[gen_init_frame-1] + 0.01 * np.random.randn(1, num_vis)

print('Generating %d frames: '%(num_gen))

for f in range(num_gen):
    initcond = np.asarray([gen_sample[s] for s in range(f,f+timesteps)]).ravel()
    # initcond = np.asarray([test_input[s+gen_init_frame] for s in range(f,f+timesteps)]).ravel()

    initframes = gen_sample[f+timesteps-1] # + 0.01 * np.random.randn(1, num_vis)

    s, h = sess.run(gen_op, feed_dict={gen_cond: np.reshape(initcond, [1,num_cond]).astype(np.float32),
                                       gen_init: initframes })

    gen_sample.append(s)
    gen_hidden.append(h)

gen_sample = np.reshape(np.asarray(gen_sample), [num_gen+timesteps,num_vis])

gen_sample = gen_sample * data_std + data_mean

fig = plt.figure(figsize=(30, 8))
plt.plot(gen_sample)
plt.savefig('%s/crbm_gen'%(snapshot_dir))
plt.close(fig)