import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import chainer
from chainer import cuda, serializers
import chainer.functions as F
from collections import defaultdict

import data
import config

conf = config.get_config()

plt.rcParams['image.cmap'] = 'gray'


def set_no_tick(ax):
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='both', which='both', length=0)


if conf.conv:
    import cnet as net
    model = net.SeqVAE(conf.w, conf.n_latent, conf.n_filter, conf.ksize,
                       conf.stride, conf.n_hidden)
else:
    import net
    model = net.SeqVAE(conf.w, conf.n_latent, conf.n_hidden, conf.n_hidden)

serializers.load_npz(conf.model_name, model)
if conf.gpu >= 0:
    cuda.get_device_from_id(conf.gpu).use()
    model.to_gpu()
xp = np if conf.gpu < 0 else cuda.cupy

dataset = data.Dataset(conf.w, conf.sw, conf.n_seq, conf.T)
dataset.initialize()

xs = dataset.sample(conf.test_batchsize, conf.test_maxt)

dic = defaultdict(list)

for i in range(conf.test_maxt):
    x_ = model.decode(model.encode(xs[i])[0], sigmoid=True)
    dic['xs_'].append(x_.data)

for i in range(conf.test_maxt):
    z = model.encode(xs[i])[0]
    dic['zs'].append(z.data)

gz = dic['zs'][0]
dic['gzs'].append(gz)
dic['gxs'].append(dic['xs_'][0])
for _ in range(conf.test_maxt-1):
    gz = model.transition(gz)[0]
    dic['gzs'].append(gz.data)
    gx = model.decode(gz, sigmoid=True)
    dic['gxs'].append(gx.data)

# original, reconstruction, latent variable
# generated latent variable, generated reconstruction
titles = ['origin', 'reconst', 'gen_reconst']
fig, axes = plt.subplots(3, conf.test_maxt, figsize=(10, 6))
for i in range(conf.test_maxt):
    im0 = axes[0, i].imshow(xs[i].reshape(conf.w, conf.w),
                            clim=(0.0, 1.0))
    im1 = axes[1, i].imshow(dic['xs_'][i].reshape(conf.w, conf.w),
                            clim=(0.0, 1.0))
    im2 = axes[2, i].imshow(dic['gxs'][i].reshape(conf.w, conf.w),
                            clim=(0.0, 1.0))
np.vectorize(lambda ax: set_no_tick(ax))(axes)
for i, im in enumerate([im0, im1, im2]):
    axes[i, 0].set_title(titles[i])
    fig.colorbar(im, ax=[axes[i, j] for j in range(conf.test_maxt)])
plt.show()

if conf.n_latent > 3:
    raise ValueError('if you wanna visualize latent variables, '
                     'dimension of latent variables should be less than 4')
cols = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#ffff33']

if conf.n_latent == 2:
    plt.figure(figsize=(10, 10))
    for i in range(4*(conf.w-1)):
        img = dataset.batch_square_with_latent(i, conf.n_plot_sample)
        img = np.array(img, dtype=np.float32)
        mu, in_var = model.encode(img)
        z = F.gaussian(mu, in_var)
        plt.plot(z.data[:, 0], z.data[:, 1], 'o', c=cols[i])
elif conf.n_latent == 3:
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(4*(conf.w-1)):
        img = dataset.batch_square_with_latent(i, conf.n_plot_sample)
        img = np.array(img, dtype=np.float32)
        mu, in_var = model.encode(img)
        z = F.gaussian(mu, in_var)
        ax.scatter(z.data[:, 0], z.data[:, 1], z.data[:, 2], 'o', c=cols[i])
plt.show()

zs = np.array(dic['zs'])
gzs = np.array(dic['gzs'])

if conf.n_latent == 2:
    plt.figure(figsize=(10, 10))
    for i in range(conf.test_maxt):
        plt.plot(zs[i, 0, 0], zs[i, 0, 1], 'o', markersize=10,
                 color=cm.Reds(float(i) / conf.test_maxt))
        plt.plot(gzs[i, 0, 0], gzs[i, 0, 1], 'o', markersize=10,
                 color=cm.Blues(float(i) / conf.test_maxt))

    plt.plot(zs[:, 0, 0], zs[:, 0, 1], linewidth=1, c='r')
    plt.plot(gzs[:, 0, 0], gzs[:, 0, 1], linewidth=1, c='b')
elif conf.n_latent == 3:
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(conf.test_maxt):
        ax.scatter(zs[i, 0, 0], zs[i, 0, 1], zs[i, 0, 2], 'o',
                   color=cm.Reds(float(i) / conf.test_maxt))
        ax.scatter(gzs[i, 0, 0], gzs[i, 0, 1], gzs[i, 0, 2], 'o',
                   color=cm.Blues(float(i) / conf.test_maxt))
    ax.plot(zs[:, 0, 0], zs[:, 0, 1], zs[:, 0, 2], linewidth=1, c='r')
    ax.plot(gzs[:, 0, 0], gzs[:, 0, 1], gzs[:, 0, 2], linewidth=1, c='b')
plt.show()
# cols = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#ffff33']
