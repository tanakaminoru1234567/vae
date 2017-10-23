import numpy as np
import cv2
import os
import pickle

import config

conf = config.Config()


class Dataset(object):

    def __init__(self, w=3, sw=1, n_seq=32, T=9):
        self.w = w
        self.sw = sw
        self.n_seq = n_seq
        self.T = T

        self.back_ground = np.zeros([w, w])
        self.square = np.ones([sw, sw])

        self.cache = 'move_square{}{}{}.pkl'.format(
                                            w, n_seq, T)

    def map_square_with_latent(self, z):
        n = conf.order[z]
        return self.map_square_with_num(n)

    def map_square_with_num(self, n):
        y, x = n // self.w, n % self.w
        img = self.back_ground.copy()
        img[y:y+self.sw, x:x+self.sw] = self.square
        return img

    def batch_square_with_latent(self, z, batch_size):
        img = self.map_square_with_latent(z)
        return np.ones([batch_size, 1, self.w, self.w]) * img

    def genetate(self):
        imgs = np.zeros([self.n_seq, self.T, self.w, self.w])
        # [number of sequence, max time step of sequence, height, width]
        for n in range(self.n_seq):
            for t in range(self.T):
                z = t % (4*(self.w-1))
                img = self.map_square_with_latent(z)
                imgs[n, t] = img
        with open(self.cache, 'wb') as f:
            pickle.dump(np.array(imgs), f)

    def load(self):
        with open(self.cache, 'rb') as f:
            imgs = pickle.load(f)
        return imgs

    def initialize(self):
        if not os.path.exists(self.cache):
            print('generate')
            self.genetate()
        self.imgs = self.load()

    def sample(self, batch_size, tsize):
        if tsize > self.T:
            raise ValueError('tsize is over T({})'.format(self.T))

        samples = np.zeros([batch_size, tsize, 1, self.w, self.w],
                           dtype=np.float32)
        for i in np.random.permutation(batch_size):
            j = i % self.n_seq
            st = np.random.randint(self.T - tsize + 1)
            samples[i, :tsize, 0] = self.imgs[j, st:st+tsize]

        return np.transpose(samples, (1, 0, 2, 3, 4))

    def movie(self):
        z = 0
        flag = True
        while(flag):
            img = self.map_square_with_latent(z)
            resized = cv2.resize(img, (conf.w_width, conf.w_width),
                                 interpolation=cv2.INTER_NEAREST)
            cv2.imshow('frame', resized)
            z = (z + 1) % (4*(self.w-1))
            if cv2.waitKey(1000) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    dataset = Dataset()
    dataset.initialize()
    print(dataset.sample(5, 2))
