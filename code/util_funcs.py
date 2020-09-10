import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from skimage.measure import block_reduce
from imageio import imread


def gabor(x, y, sig, theta, gamma, lamb):

    x0 = x * np.cos(theta) + y * np.sin(theta)
    y0 = -x * np.sin(theta) + y * np.cos(theta)

    g = np.exp(-(x0**2 + gamma**2 * y0**2) /
               (2 * sig**2)) * np.cos(2 * np.pi * x0 / lamb)

    return g


def make_s1_filter_bank():

    # define layer 1 filter bank from Serre (2007)
    s1_size = np.arange(7, 39, 2)
    s1_sig = np.array([
        2.8, 3.6, 4.5, 5.4, 6.3, 7.3, 8.2, 9.2, 10.2, 11.3, 12.3, 13.4, 14.6,
        15.8, 17.0, 18.2
    ])
    s1_lamb = np.array([
        3.5, 4.6, 5.6, 6.8, 7.9, 9.1, 10.3, 11.5, 12.7, 14.1, 15.4, 16.8, 18.2,
        19.7, 21.2, 22.8
    ])
    s1_orientations = np.array(
        [x * np.pi / 180 for x in (0.0, 45.0, 90.0, 135.0)])

    filter_bank = []
    for j in range(4):
        for i in range(16):
            step = 1
            x = np.arange(-s1_size[i], s1_size[i], step)
            y = np.arange(-s1_size[i], s1_size[i], step)
            xv, yv = np.meshgrid(x, y)
            g = gabor(xv, yv, s1_sig[i], s1_orientations[j], 0.3, s1_lamb[i])
            filter_bank.append(g)

    return filter_bank


def get_s1(input_img, filter_bank, window_size):

    # roll over the image with a sliding window, and
    # for each window, compute the convolution.
    stride = window_size
    num_windows_x = ((input_img.shape[0] - window_size) // stride) + 1
    num_windows_y = ((input_img.shape[1] - window_size) // stride) + 1

    filt_img_rec = np.zeros((len(filter_bank), num_windows_x, num_windows_y,
                             window_size, window_size))

    for k in range(len(filter_bank)):

        filt = filter_bank[k]

        for i in range(num_windows_x):
            for j in range(num_windows_y):

                # select pixels from image to be filtered with the current window
                x_ind = np.arange(i * stride, i * stride + window_size, 1)
                y_ind = np.arange(j * stride, j * stride + window_size, 1)
                xv_ind, yv_ind = np.meshgrid(x_ind, y_ind)
                sub_img = input_img[yv_ind, xv_ind]

                # apply gabor filter
                filt_img = signal.convolve2d(sub_img,
                                             filt,
                                             boundary='fill',
                                             mode='same')

                filt_img_rec[k, i, j, :, :] = filt_img

    return filt_img_rec


def get_c1(s1):

    # define max pooling kernel sizes and stride lengths
    ns = np.arange(8, 23, 2)
    ds = np.array([4, 5, 6, 7, 8, 9, 10, 11])

    # repeat ns and ds four times (once for each orientation)
    ns = np.tile(ns, 4)
    ds = np.tile(ds, 4)

    # four orientations x 8 bands x 4, 4, ?, ?
    # We will handle the unknown dimension sizes via zero padding later
    c1_shape = np.array(s1.shape)
    c1_shape[0] = c1_shape[0] // 2
    c1 = np.zeros(c1_shape)

    for k in range(0, s1.shape[0], 2):
        for i in range(s1.shape[1]):
            for j in range(s1.shape[2]):

                ss1 = s1[k, i, j, :, :]
                ss2 = s1[k + 1, i, j, :, :]

                band = k // 2

                num_win_x = ((ss1.shape[0] - ns[band]) // ds[band]) + 1
                num_win_y = ((ss1.shape[1] - ns[band]) // ds[band]) + 1
                num_win = num_win_x * num_win_y

                cc1 = np.zeros((num_win_x, num_win_y))
                for winx in range(num_win_x):
                    for winy in range(num_win_y):

                        x_ind = np.arange(winx * ds[band],
                                          winx * ds[band] + ns[band], 1)
                        y_ind = np.arange(winy * ds[band],
                                          winy * ds[band] + ns[band], 1)
                        xv_ind, yv_ind = np.meshgrid(x_ind, y_ind)
                        sss1 = ss1[yv_ind, xv_ind]
                        sss2 = ss2[yv_ind, xv_ind]

                        sss1_max = np.max(sss1)
                        sss2_max = np.max(sss2)

                        cc1[winx, winy] = np.max((sss1_max, sss2_max))

                pad = np.array(c1.shape[-2:]) - np.array(cc1.shape[-2:])

                cc1_padded = np.pad(cc1, ((np.ceil(pad).astype(int)[0], 0),
                                          (np.ceil(pad).astype(int)[1], 0)),
                                    'constant',
                                    constant_values=np.nan)

                c1[band, i, j, :, :] = cc1_padded

    return c1


def make_s2_filter_bank(c1_in, filter_bank, window_size):

    prototype_list = []

    rf_size = 3
    n_prototype = 5
    n_scale = c1_in.shape[0] // 4
    band_scales = np.arange(8, 23, 2)

    img_file_list = os.listdir("../images")
    for i in range(n_prototype):

        img_file = np.random.choice(img_file_list)
        img = imread('../images/' + img_file)

        s1 = get_s1(img, filter_bank, window_size)
        c1 = get_c1(s1)

        # At the ith image presentation, one unit at a particular position and
        # scale is selected (at random) from the ith feature-map and is
        # imprinted. That is, the unit stores in its synaptic weights w, the
        # current pattern of activity from its afferent inputs, in response to
        # the part of the natural image i that falls within its receptive
        # field.

        # remove nan padding
        prototype_list_2 = []
        for j in range(4):

            scale_ind = np.random.randint(0, n_scale)
            scale = band_scales[scale_ind]

            retx = np.random.randint(0, c1.shape[1])
            rety = np.random.randint(0, c1.shape[2])
            cc1 = c1[j * n_scale + scale_ind, retx, rety, :, :]

            keep = ~np.all(np.isnan(cc1), 0)
            cc1 = cc1[keep, :]
            keep = np.all(~np.isnan(cc1), 0)
            cc1 = cc1[:, keep]

            x0 = np.random.randint(0, cc1.shape[0] - rf_size)
            y0 = np.random.randint(0, cc1.shape[1] - rf_size)
            x = np.arange(x0, x0 + rf_size)
            y = np.arange(y0, y0 + rf_size)
            xv_ind, yv_ind = np.meshgrid(x, y)

            prototype_list_2.append(cc1[xv_ind, yv_ind])

        prototype = np.stack(prototype_list_2)
        prototype = np.mean(prototype, 0) / np.linalg.norm(prototype)

        prototype_list.append(prototype)

    return prototype_list


def get_s2(c1, filter_bank, window_size):

    for ii in range(c1.shape[0]):
        for jj in range(c1.shape[1]):
            for kk in range(c1.shape[2]):

                cc1 = c1[ii, jj, kk, :, :]
                keep = ~np.all(np.isnan(cc1), 0)
                cc1 = cc1[keep, :]
                keep = np.all(~np.isnan(cc1), 0)
                cc1 = cc1[:, keep]

                # roll over the image with a sliding window, and
                # for each window, compute the convolution.
                stride = window_size
                num_windows_x = ((cc1.shape[0] - window_size) // stride) + 1
                num_windows_y = ((cc1.shape[1] - window_size) // stride) + 1

                filt_img_rec = np.zeros(
                    (len(filter_bank), num_windows_x, num_windows_y,
                     window_size, window_size))

                for k in range(len(filter_bank)):

                    filt = filter_bank[k]

                    for i in range(num_windows_x):
                        for j in range(num_windows_y):

                            # select pixels from image to be filtered with the
                            # current window
                            x_ind = np.arange(i * stride,
                                              i * stride + window_size, 1)
                            y_ind = np.arange(j * stride,
                                              j * stride + window_size, 1)
                            xv_ind, yv_ind = np.meshgrid(x_ind, y_ind)
                            sub_img = cc1[yv_ind, xv_ind]

                            # apply filter
                            filt_img = signal.convolve2d(sub_img,
                                                         filt,
                                                         boundary='fill',
                                                         mode='same')

                            filt_img_rec[k, i, j, :, :] = filt_img

    return filt_img_rec


def get_c2(s2):
    pass


def inspect_s1(s1):

    for filt in range(0, 4, 1):
        fig, ax = plt.subplots(4, 4, squeeze=False)
        for i in range(4):
            for j in range(4):
                ax[i, j].imshow(s1[filt, i, j, :, :], cmap='gray')
                ax[i, j].axes.get_xaxis().set_visible(False)
                ax[i, j].axes.get_yaxis().set_visible(False)
        plt.show()


def inspect_c1(c1):

    for filt in range(0, 4, 1):
        fig, ax = plt.subplots(4, 4, squeeze=False)
        for i in range(4):
            for j in range(4):
                x = c1[filt, i, j, :, :]
                ax[i, j].imshow(x, cmap='gray')
                ax[i, j].axes.get_xaxis().set_visible(False)
                ax[i, j].axes.get_yaxis().set_visible(False)
                xmin = np.where(~np.isnan(x[-1, :]))[0][0]
                xmax = np.where(~np.isnan(x[-1, :]))[0][-1]
                ymin = np.where(~np.isnan(x[:, -1]))[0][0]
                ymax = np.where(~np.isnan(x[:, -1]))[0][-1]
                ax[i, j].set_xlim([xmin, xmax])
                ax[i, j].set_ylim([ymin, ymax])
        plt.show()
