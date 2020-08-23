#%% import and create folders
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import misc
from copy import deepcopy
import os
import math


#%% define gabor function
def gabor(x, y, sig, theta, gamma, lamb):

    x0 = x * np.cos(theta) + y * np.sin(theta)
    y0 = -x * np.sin(theta) + y * np.cos(theta)

    g = np.exp(-(x0**2 + gamma**2 * y0**2) /
               (2 * sig**2)) * np.cos(2 * np.pi * x0 / lamb)

    return g


#%% define filter bank function and generate filter_bank
def make_s1_filter_bank():

    # define layer 1 filter bank from Serre (2007)
    s1_size = [x for x in range(7, 39, 2)]
    s1_sig = [
        2.8, 3.6, 4.5, 5.4, 6.3, 7.3, 8.2, 9.2, 10.2, 11.3, 12.3, 13.4, 14.6,
        15.8, 17.0, 18.2
    ]
    s1_lamb = [
        3.5, 4.6, 5.6, 6.8, 7.9, 9.1, 10.3, 11.5, 12.7, 14.1, 15.4, 16.8, 18.2,
        19.7, 21.2, 22.8
    ]
    s1_orientations = [x * np.pi / 180 for x in (0.0, 45.0, 90.0, 135.0)]

    filter_bank = []
    for i in range(16):
        for j in range(4):
            step = 1
            x = np.arange(-s1_size[i], s1_size[i], step)
            y = np.arange(-s1_size[i], s1_size[i], step)
            xv, yv = np.meshgrid(x, y)
            g = gabor(xv, yv, s1_sig[i], s1_orientations[j], 0.3, s1_lamb[i])
            filter_bank.append(g)

    # inspect s1 filters
    # fig, ax = plt.subplots(nrows=16, ncols=4, figsize=(2, 8))
    for i in range(16):
        for j in range(4):
            step = 1
            # need to create the gabors on the largest needed grid in order to
            # see differences in spatial frequency when we plot
            x = np.arange(-s1_size[-1], s1_size[-1], step)
            y = np.arange(-s1_size[-1], s1_size[-1], step)
            xv, yv = np.meshgrid(x, y)
            g = gabor(xv, yv, s1_sig[i], s1_orientations[j], 0.3, s1_lamb[i])
            # ax[i, j].imshow(g, cmap='gray')
            # ax[i, j].axes.get_xaxis().set_visible(False)
            # ax[i, j].axes.get_yaxis().set_visible(False)

    #create figures folder if it doesn't exist
    if not os.path.exists('fig'):
        os.makedirs('fig')

    #create and save the plot
    # plt.tight_layout()
    # plt.savefig(os.getcwd() + "/fig/gabor_array.png")
    #plt.show()

    return filter_bank


# create s1 filter bank
filter_bank = make_s1_filter_bank()

#%% apply filters and plot results
# create figures folder if it doesn't exist
fig_num = 0  # this is to define the filename. 0 is for the test image
if not os.path.exists('fig'):
    os.makedirs('fig')
if not os.path.exists('fig/filtered'):
    os.makedirs('fig/filtered')

# grab and save a test image
ascent = misc.ascent()

# fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False, figsize=(4, 10))
# ax[0, 0].imshow(ascent, cmap='gray')
# plt.savefig(os.getcwd() + "/fig/filtered/" + str(fig_num) + "_filtered.png")
# ax[0, 0].show()

# roll over the image with a sliding window, and
# for each window, compute the convolution.
window_size = 512 // 4
stride = window_size
num_windows_x = ((ascent.shape[0] - window_size) // stride) + 1
num_windows_y = ((ascent.shape[1] - window_size) // stride) + 1

sub_img_filt_rec = np.zeros((len(filter_bank), num_windows_x,
                                num_windows_y, window_size, window_size))

# for filt in filter_bank:
for k in range(len(filter_bank)):

    filt = filter_bank[k]

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

            ax[j, i].imshow(sub_img, cmap='gray')

            # apply gabor filter
            sub_img_filt = signal.convolve2d(sub_img,
                                             filt,
                                             boundary='fill',
                                             mode='same')

            sub_img_filt_rec[k, i, j, :, :] = sub_img_filt

            ax[j, i].imshow(sub_img_filt, cmap='gray')
            ax[j, i].axes.get_xaxis().set_visible(False)
            ax[j, i].axes.get_yaxis().set_visible(False)

    # create and save the files
    fig_num = fig_num + 1
    fig_sz_range = [x for x in range(7, 39, 2)]
    fig_sz = str(fig_sz_range[(math.ceil(((fig_num / 4) - 0.1) - 1))])
    fig_ang_range = [0, 45, 90, 135]
    fig_ang = str(fig_ang_range[(fig_num - 1) - (math.floor(fig_num / 4)) * 4])
    fig_name = (str(fig_num) + '_' + fig_sz + '_' + fig_ang + '_')
    plt.tight_layout()
    plt.savefig(os.getcwd() + "/fig/filtered/filtered_" + fig_name + ".png")
    # plt.show()
    plt.close('all')

    # TODO: fix band iteration
    # TODO: basically, matched orientations are every 4 elements
    s1_size = np.arange(7, 39, 2)
    pool_1 = s1_size[0::2]
    pool_2 = s1_size[1::2]

    ns = np.arange(8, 23, 2)
    ds = np.array([4, 5, 6, 7, 8, 9, 10, 11])
    for band in range(pool_1.shape[0]):
        for i in range(sub_img_filt_rec.shape[1]):
            for j in range(sub_img_filt_rec.shape[2]):
                s1 = sub_img_filt_rec[band, i, j, :, :]
                s2 = sub_img_filt_rec[band + 1, i, j, :, :]

    # sub_max = []
    # for band in range(pool_1.shape[0]):
    #     sub_max_band = []
    #     s1 = sub_img_filt_rec[band]
    #     s2 = sub_img_filt_rec[band + 1]
    #     num_win_x = ((s1.shape[0] - ns[band]) // ds[band]) + 1
    #     num_win_y = ((s1.shape[1] - ns[band]) // ds[band]) + 1
    #     num_win = num_win_x * num_win_y
    #     for winx in range(num_win_x):
    #         for winy in range(num_win_y):
    #             x_ind = np.arange(i * ds[band], i * ds[band] + ns[band], 1)
    #             y_ind = np.arange(j * ds[band], j * ds[band] + ns[band], 1)
    #             xv_ind, yv_ind = np.meshgrid(x_ind, y_ind)
    #             sub_s1 = s1[yv_ind, xv_ind]
    #             sub_s2 = s2[yv_ind, xv_ind]
    #             sub_s1_max = np.max(sub_s1)
    #             sub_s2_max = np.max(sub_s2)
    #             sub_max_band.append(np.max([sub_s1_max, sub_s2_max]))
    #     sub_max.append(sub_max_band)
