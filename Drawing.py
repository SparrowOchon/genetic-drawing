import cv2
import numpy as np
from time import time
import matplotlib.pyplot as plt

from Gene import DNA


class GeneticDrawing:
    def __init__(self, img_path, seed=0, brush_range=[[0.1, 0.3], [0.3, 0.7]]):
        self.original_img = cv2.imread(img_path)
        self.img_grey = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
        self.img_grads = self.__img_gradient(self.img_grey)
        self.img_dna = None
        self.seed = seed
        self.brush_range = brush_range
        self.sampling_mask = None

        # start with an empty black img
        self.img_buffer = [
            np.zeros((self.img_grey.shape[0], self.img_grey.shape[1]), np.uint8)
        ]

    def generate(
        self, stages=10, generations=100, brush_strokes=10, show_progress_imgs=False
    ):
        for s in range(stages):
            # initialize new DNA
            if self.sampling_mask is not None:
                sampling_mask = self.sampling_mask
            else:
                sampling_mask = self.__create_sampling_mask(s, stages)
            self.img_dna = DNA(
                self.img_grey.shape,
                self.img_grads,
                self.__calc_brush_range(s, stages),
                canvas=self.img_buffer[-1],
                sampling_mask=sampling_mask,
            )
            self.img_dna.randomize_dna(
                self.img_grey, brush_strokes, self.seed + time() + s
            )
            # evolve DNA
            for g in range(generations):
                self.img_dna.evolve_dna(self.img_grey, self.seed + time() + g)
                # clear_output(wait=True)
                print("Stage ", s + 1, ". Generation ", g + 1, "/", generations)
                if show_progress_imgs is True:
                    # plt.imshow(sampling_mask, cmap='gray')
                    plt.imshow(self.img_dna.cached_image, cmap="gray")
                    plt.show()
            self.img_buffer.append(self.img_dna.cached_image)
        return self.img_dna.cached_image

    def __calc_brush_range(self, stage, total_stages):
        return [
            self.__calc_brush_size(self.brush_range[0], stage, total_stages),
            self.__calc_brush_size(self.brush_range[1], stage, total_stages),
        ]

    def __create_sampling_mask(self, s, stages):
        percent = 0.2
        start_stage = int(stages * percent)
        sampling_mask = None
        if s >= start_stage:
            t = (
                1.0 - (s - start_stage) / max(stages - start_stage - 1, 1)
            ) * 0.25 + 0.005
            sampling_mask = self.__calc_sampling_mask(t)
        return sampling_mask

    # we'd like to "guide" the brushtrokes along the image gradient direction, if such direction has large magnitude
    # in places of low magnitude, we allow for more deviation from the direction.
    # this function precalculates angles and their magnitudes for later use inside DNA class

    def __img_gradient(self, img):
        # convert to 0 to 1 float representation
        img = np.float32(img) / 255.0
        # Calculate gradient
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
        # Python Calculate gradient magnitude and direction ( in degrees )
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        # normalize magnitudes
        mag /= np.max(mag)
        # lower contrast
        mag = np.power(mag, 0.3)
        return mag, angle

    def __calc_sampling_mask(self, blur_percent):
        img = np.copy(self.img_grey)
        # Calculate gradient
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
        # Python Calculate gradient magnitude and direction ( in degrees )
        mag, _ = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        # calculate blur level
        w = img.shape[0] * blur_percent
        if w > 1:
            mag = cv2.GaussianBlur(mag, (0, 0), w, cv2.BORDER_DEFAULT)
        # ensure range from 0-255 (mostly for visual debugging, since in sampling we will renormalize it anyway)
        scale = 255.0 / mag.max()
        return mag * scale

    def __calc_brush_size(self, brange, stage, total_stages):
        bmin = brange[0]
        bmax = brange[1]
        t = stage / max(total_stages - 1, 1)
        return (bmax - bmin) * (-t * t + 1) + bmin
