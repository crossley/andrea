import numpy as np
import matplotlib.pyplot as plt


def gabor(x, y, sig, theta, gamma, lamb):

    x0 = x * np.cos(theta) + y * np.sin(theta)
    y0 = -x * np.sin(theta) + y * np.cos(theta)

    g = np.exp(-(x0**2 + gamma**2 * y0**2) /
               (2 * sig**2)) * np.cos(2 * np.pi * x0 / lamb)

    return g


x = 0
y = 0
sig = 1
theta = 125 * np.pi / 180
size = 4
gamma = 1
lamb = 0.5

step = 0.1
x = np.arange(-size, size, step)
y = np.arange(-size, size, step)
xv, yv = np.meshgrid(x, y)
g = gabor(xv, yv, sig, theta, gamma, lamb)

plt.imshow(g, cmap='gray')
plt.show()

## define layer 1 filter bank
theta = 45
lamb = 0.5
g1 = gabor(xv, yv, sig, theta, gamma, lamb)

theta = -45
lamb = 2.0
g2 = gabor(xv, yv, sig, theta, gamma, lamb)

filter_list = [g1, g2]

from scipy import signal
from scipy import misc

ascent = misc.ascent()

plt.imshow(ascent, cmap='gray')
plt.show()

s = signal.convolve2d(ascent, g, boundary='fill', mode='valid')

plt.imshow(s, cmap='gray')
plt.show()

# roll over the image with a sliding window, and
# for each window, compute the convolution.
window_size = 512 // 2
stride = window_size

num_windows_x = ((ascent.shape[0] - window_size) // stride) + 1
num_windows_y = ((ascent.shape[1] - window_size) // stride) + 1
num_windows = num_windows_x * num_windows_y

for filt in filter_list:
    fig, ax = plt.subplots(nrows=num_windows_x,
                       ncols=num_windows_y,
                       figsize=(6, 6))
    for i in range(num_windows_x):
        for j in range(num_windows_y):
            # select pixels from image to be filtered with the current window
            x_ind = np.arange(i * stride, i * stride + window_size, 1)
            y_ind = np.arange(j * stride, j * stride + window_size, 1)
            xv_ind, yv_ind = np.meshgrid(x_ind, y_ind)
            sub_img = ascent[yv_ind, xv_ind]
            # ax[j, i].imshow(sub_img, cmap='gray')

            # apply gabor filter
            sub_img_filt = signal.convolve2d(sub_img,
                                             filt,
                                             boundary='fill',
                                             mode='valid')

            ax[j, i].imshow(sub_img_filt, cmap='gray')

    plt.show()
