import numpy as np
import chainer
from chainer import cuda, optimizers, serializers
import os

import data
import config

if not os.path.exists('data/'):
    os.mkdir('data/')
if not os.path.exists('model/'):
    os.mkdir('model/')


conf = config.get_config()
conf.ww = conf.sw * conf.w

if conf.conv:
    import cnet as net
    model = net.SeqVAE(conf.ww, conf.n_latent, conf.n_filter, conf.ksize,
                       conf.stride, conf.n_hidden)
else:
    import net
    model = net.SeqVAE(conf.ww, conf.n_latent, conf.n_hidden, conf.n_hidden)

if conf.gpu >= 0:
    cuda.get_device_from_id(conf.gpu).use()
    model.to_gpu()
xp = np if conf.gpu < 0 else cuda.cupy

optimizer = optimizers.Adam()
optimizer.setup(model)

dataset = data.Dataset(conf.w, conf.sw, conf.n_seq, conf.T)
dataset.initialize()

for i in range(conf.n_iter):
    xs = dataset.sample(conf.batchsize, conf.tsize)
    xs = chainer.Variable(xs)
    optimizer.update(model.get_loss_func(k=1, c=1.0), xs)
    if (i+1) % conf.print_freq == 0:
        print(('iteration:{:>4}, '
               'L_x:{:.3f}, L_z:{:.3f}, '
               'L_nx:{:.3f}, L_nz:{:.3f}, '
               'total_loss:{:.3f}'
               .format(i+1,
                       float(model.l_x[0].data), float(model.l_z[0].data),
                       float(model.l_x[1].data), float(model.l_z[1].data),
                       float(model.loss.data))))

serializers.save_npz(conf.model_name, model)
