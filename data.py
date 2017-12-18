import numpy as np
import cv2
import os
import pickle

import config

conf = config.get_config()


class Dataset(object):

    def __init__(self, w=3, sw=1, n_seq=32, T=9, random=True):
        self.w = w
        self.sw = sw
        self.ww = w * sw
        self.n_seq = n_seq
        self.T = T

        self.back_ground = np.zeros([self.ww, self.ww])
        self.square = np.ones([sw, sw])

        if random:
            r = 'r'
            self.gen_traj = self.generate_random_trajectory
        else:
            r = ''
            self.gen_traj = self.generate_trajectory

        self.cache = 'data/move_square{}{}{}{}{}.pkl'.format(
                                            w, sw, n_seq, T, r)

    def map_square_with_latent(self, z):
        n = conf.order[z]
        return self.map_square_with_num(n)

    def map_square_with_num(self, n):
        y, x = n // self.w, n % self.w
        y, x = self.sw * y, self.sw * x
        img = self.back_ground.copy()
        img[y:y+self.sw, x:x+self.sw] = self.square
        return img

    def batch_square_with_latent(self, z, batch_size):
        img = self.map_square_with_latent(z)
        return np.ones([batch_size, 1, self.ww, self.ww]) * img

    def genetate(self):
        imgs = np.zeros([self.n_seq, self.T, self.ww, self.ww])
        # [number of sequence, max time step of sequence, height, width]
        for n in range(self.n_seq):
            imgs[n] = self.gen_traj(self.T)
        with open(self.cache, 'wb') as f:
            pickle.dump(np.array(imgs), f)

    def generate_trajectory(self, T):
        traj = np.zeros([T, self.ww, self.ww])
        for t in range(T):
            z = t % (4*(self.w-1))
            img = self.map_square_with_latent(z)
            traj[t] = img
        return traj

    def generate_random_trajectory(self, T):
        traj = np.zeros([T, self.ww, self.ww])
        z = 0
        for t in range(T):
            z %= (4*(self.w-1))
            img = self.map_square_with_latent(z)
            traj[t] = img

            if z == 2:
                z += np.random.choice([-2, 0], p=[0.5, 0.5])
            z += 1
        return traj

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

        samples = np.zeros([batch_size, tsize, 1, self.ww, self.ww],
                           dtype=np.float32)
        for i, j in enumerate(np.random.permutation(batch_size)):
            j %= self.n_seq
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
    dataset = Dataset(T=4)
    dataset.initialize()
    # print(dataset.sample(10, 4))
    # print(dataset.sample(5, 2))
    # dataset.movie()
