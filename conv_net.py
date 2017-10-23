import chainer
import chainer.functions as F
import chainer.links as L
from chainer.functions.loss import vae
from chainer.functions.math import sum, exponential


class SeqVAE(chainer.Chain):

    def __init__(self, w_input, n_latent, n_filter, n_h2):
        super(SeqVAE, self).__init__()
        self.w_input = w_input
        self.n_filter = n_filter
        self.w_hidden = w_input - 1
        n_hidden = n_filter * self.w_hidden * self.w_hidden
        with self.init_scope():
            # encoder
            self.le1 = L.Convolution2D(1, n_filter, ksize=2)
            self.le2 = L.Linear(n_hidden, 2 * n_latent)
            # decoder
            self.ld1 = L.Linear(n_latent, n_hidden)
            self.ld2 = L.Deconvolution2D(n_filter, n_filter, ksize=2)
            self.ld3 = L.Convolution2D(n_filter, 1, ksize=1)
            # transion
            self.lt1 = L.Linear(n_latent, n_h2)
            self.lt2 = L.Linear(n_h2, 2 * n_latent)

    def encode(self, x):
        # h1 = F.leaky_relu(self.le1(x))
        h1 = F.relu(self.le1(x))
        h2 = self.le2(h1)
        mu, in_var = F.split_axis(h2, 2, axis=1)
        return mu, in_var

    def decode(self, z, sigmoid=False):
        # h1 = F.leaky_relu(self.ld1(z))
        h1 = F.relu(self.ld1(z))
        h1 = F.reshape(h1, (-1, self.n_filter, self.w_hidden, self.w_hidden))
        h2 = F.relu(self.ld2(h1))
        h3 = self.ld3(h2)
        if sigmoid:
            return F.sigmoid(h3)
        else:
            return h3

    def transition(self, z):
        # h1 = F.leaky_relu(self.lt1(z))
        h1 = F.relu(self.lt1(z))
        h2 = self.lt2(h1)
        mu, in_var = F.split_axis(h2, 2, axis=1)
        return mu, in_var

    def get_loss_func(self, k=1, c=0.8):

        def lf(xs):
            T, batchsize = xs.shape[:2]
            self.loss = 0.
            self.losses = [0.] * T
            self.l_x = [0.] * T
            self.l_z = [0.] * T
            prev_dist = None

            for t in range(T):
                # reconstruction loss x
                l_x, dist = self.reconstruction_loss(xs[t], k)

                # D_KL [ q || p ]
                if t == 0:
                    l_z = vae.gaussian_kl_divergence(dist.mu, dist.ln_var) \
                                                            / batchsize
                else:
                    l_z = 0.
                    for z in prev_dist.samples:
                        mu, ln_var = self.transition(z)
                        kl = self.gaussian_kl_divergence(dist.mu, dist.ln_var,
                                                         mu, ln_var)
                        # k == len(prev_dist.samples)
                        l_z += kl / (k * batchsize)

                self.l_x[t] = l_x
                self.l_z[t] = l_z
                self.losses[t] = l_x + c * l_z
                self.loss += self.losses[t]
                prev_dist = dist

            return self.loss
        return lf

    def reconstruction_loss(self, x, k):
        batchsize = len(x)
        mu, ln_var = self.encode(x)
        samples = []
        loss = 0.
        for _ in range(k):
            z = F.gaussian(mu, ln_var)
            samples.append(z)
            loss += F.bernoulli_nll(x, self.decode(z)) / (k * batchsize)
        dist = Distribution(mu, ln_var, samples)
        return loss, dist

    def gaussian_kl_divergence(self, mu1, ln_var1, mu2, ln_var2):
        # D_KL [ N(z ; mu1, var1) || N(z; mu2, var2) ]
        var1 = exponential.exp(ln_var1)
        inv_var2 = exponential.exp(-ln_var2)
        mu_diff = mu2 - mu1
        term1 = (var1 + mu_diff * mu_diff) * inv_var2
        loss = (term1 - ln_var1 + ln_var2 - 1.) * 0.5
        return sum.sum(loss)


class Distribution(object):
    def __init__(self, mu, ln_var, samples):
        self.mu = mu
        self.ln_var = ln_var
        self.samples = samples
