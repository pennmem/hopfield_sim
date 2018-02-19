import numpy as np
import PIL.Image
import math
import matplotlib.pyplot as plt
import os
from glob import glob
import time
import random
from collections import OrderedDict

from settings import settings

import logging
logging.basicConfig()
log = logging.getLogger()
log.setLevel(logging.DEBUG)

class Image(object):

    VALID_EXTENSIONS = '.png', '.jpg', '.jpeg'

    def __init__(self, filename, config=settings):
        self.config = config

        self.filename = filename
        self.image = PIL.Image.open(filename, mode="r").convert("RGBA")


        self._vector = self._calculate_vector()
        self._weights = self._calculate_weights()

        log.debug("Image {} created".format(filename))

    def basename(self):
        return os.path.basename(self.filename)

    def resize(self):
        self._vector = self._calculate_vector()
        self._weights = self._calculate_weights()

    def _calculate_vector(self):
        resized_image = self.image.resize(self.config.size[::-1])

        color_mat = np.asarray(resized_image)

        bw_image = resized_image.convert("L")

        matrix = np.array(np.asarray(bw_image))

        matrix[color_mat[:,:,3] == 0] = 255

        vector = np.reshape(matrix, self.config.size[0] * self.config.size[1])

        vector = (vector > np.mean(vector)).astype(self.config.weight_dtype)
        vector[vector!=1] = -1

        return vector

    def _calculate_weights(self):
        log.debug("Calculating weights")
        weights = np.outer(self._vector, self._vector)
        log.debug("Weights calculated")
        return weights

    @property
    def vector(self):
        return self._vector

    @property
    def matrix(self):
        return self._vector.reshape(self.config.size)

    @property
    def weights(self):
        return self._weights

class HopfieldException(Exception):
    pass

class HopfieldNetwork(object):

    def __init__(self, config=settings):
        self.config = config
        self.weight_size = config.size[0] * config.size[1]
        self.weights = np.zeros(shape=(self.weight_size, self.weight_size), dtype=self.config.weight_dtype)

        self.images    = OrderedDict()
        self.strengths = OrderedDict()

        self.state = np.zeros(shape = (self.weight_size), dtype=bool)

        self.iterations = 0

    @property
    def state_mat(self):
        return self.state.reshape(self.config.size)

    def state_matches(self, percent):
        for image in self.images.values():
            non_match = np.count_nonzero(self.state == image.vector)
            if float(non_match) / self.weight_size > percent:
                return True
        return False



    def reset_state(self, ids, ratios, noise):

        images = [self.get_image(id) for id in ids]

        self.iterations = 0
        self.initialize_state(images, ratios, noise)

    def step(self):
        if self.config.synchronous:
            self.synchronous_step()
        else:
            self.asynch_step()

    @classmethod
    def resolve_state(cls, state, temperature = None):
        if temperature is None or temperature == 0:
            state[state > 0] = 1

            state[state < 0] = -1

            mask = state == 0
            state[mask] = np.random.choice([-1, 1], np.count_nonzero(mask), replace=True)
        else:
            p = np.power((1+np.exp(-1./temperature*state)), -1)
            rand_threshold = np.random.random((p.shape))
            state[p > rand_threshold] = 1
            state[p <= rand_threshold] = -1
        return state

    def synchronous_step(self):
        state_flat = self.state
        new_state = np.dot(self.weights, state_flat)
        self.resolve_state(new_state, self.config.temperature)
        self.state = new_state
        self.iterations += 1

    def asynch_step(self):
        state_flat = self.state
        t = time.time()
        for _ in range(self.config.async_speed):
            index = random.randint(0, state_flat.shape[0]-1)
            #new_val = np.sum(np.inner(self.weights[:,index], state_flat))
            new_val = np.einsum('i,i->', self.weights[:,index], state_flat)
            state_flat[index] = self.resolve_state(np.array(new_val))
        log.debug("async took {}".format(time.time()-t))
        self.iterations += self.config.async_speed
        self.state = state_flat

    def _generate_initial_state(self, images, ratios, noise_percentage):

        if len(images) != len(ratios):
            raise HopfieldException("images and ratios must have same length, "
                                    "not {} and {}".format(len(images), len(ratios)))

        limits = [0]
        for ratio in ratios:
            limits.append(limits[-1] + ratio)

        limits = np.array(limits)
        max_limit = max(limits)

        log.debug("Creating state")

        rands = np.random.rand(self.weight_size) * max_limit

        state = np.ones(shape = self.weight_size) * -1

        for i, (min_lim, max_lim) in enumerate(zip(limits, limits[1:])):
            mask = np.logical_and(rands>=min_lim, rands<max_lim)
            state[mask] = images[i].vector[mask]

        rands = np.random.rand(self.weight_size)
        mask = rands < noise_percentage
        if self.config.flipped_noise:
            state[mask] = np.logical_not(state[mask]==1)
            state[state == 0 ]= -1
        else:
            n_random = np.count_nonzero(mask)
            log.debug("Randomizing {} pixels".format(n_random))
            new_pixels = (np.random.rand(self.weight_size)[mask] > .5).astype('int8')
            new_pixels[new_pixels == 0] = -1
            state[mask] = new_pixels

        return state

    def initialize_state(self, images, ratios, noise_percentage):
        self.state = self._generate_initial_state(images, ratios, noise_percentage)

    def get_image(self, id):
        if isinstance(id, int):
            return self.images.values()[id]
        else:
            return self.images[id]

    def set_strength(self, id, strength):
        orig_strengths = self.get_strength(id)
        if isinstance(id, int):
            key = self.strengths.keys()[id]
            self.strengths[key] = strength
        else:
            self.strengths[id] = strength
        strength_diff = strength - orig_strengths
        if strength_diff != 0:
            log.debug("Changing strength of {} by {}".format(id, strength_diff))
            image = self.get_image(id)
            self.weights += image.weights * strength_diff

    def get_strength(self, id):
        if isinstance(id, int):
            return self.strengths.values()[id]
        else:
            return self.strengths[id]

    def get_images(self):
        return self.images.values()

    def add_directory(self, directory):
        for extension in Image.VALID_EXTENSIONS:
            files = glob(os.path.join(directory, '*'+extension))
            for file in files:
                self.add_image(file, 1)

    def add_image(self, filename, strength=1):
        log.debug("Adding image {}".format(filename))
        if filename in self.images:
            raise HopfieldException("Image {} already exists in library".format(filename))

        image = Image(filename, self.config)
        self.images[filename] = image
        self.strengths[filename] = strength
        log.debug("Appending to weights")
        self.weights += image.weights * strength
        log.debug("Image added")
        self.weights[::self.weights.shape[0]+1] = 0

    def clear(self):
        self.weights = np.zeros(self.weights.shape)
        self.images.clear()
        self.strengths.clear()
        self.state = np.zeros(self.state.shape)
        self.iterations = 0

    def remove_image(self, filename):
        if filename not in self.images:
            raise HopfieldException("Image {} not in library".format(filename))

        image = self.images[filename]
        strength = self.strengths[filename]
        self.weights -= image.weights * strength

        del self.images[filename]
        del self.strengths[filename]

    def contains_image(self, filename):
        return filename in self.images

    def recalculate(self):
        self.weight_size = self.config.size[0] * self.config.size[1]
        self.weights = np.zeros(shape=(self.weight_size, self.weight_size), dtype=self.config.weight_dtype)
        self.state = np.zeros(shape=self.weight_size, dtype=bool)

        filenames = self.images.keys()
        strengths = self.strengths.values()
        self.clear()
        for filename, strength in zip(filenames, strengths):
            self.add_image(filename, strength)

if __name__ == "__main__":
    hopfield_network = HopfieldNetwork()

    hopfield_network.add_image('images/brain.jpg')
    hopfield_network.add_image('images/earth.png')
    hopfield_network.add_image('images/lightbulb.png')
    hopfield_network.add_image('images/ufo.jpg')
    image1 = hopfield_network.get_image(0)
    image2 = hopfield_network.get_image(1)

    hopfield_network.initialize_state([image1, image2], [5, .5], .1)

    fig, ax = plt.subplots()
    shown = plt.imshow(hopfield_network.state)

    def update(data):
        hopfield_network.step()
        shown.set_array(hopfield_network.state)
        return shown,

    import matplotlib.animation as animation
    ani = animation.FuncAnimation(fig, update, interval=10)

    plt.show()
