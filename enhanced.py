import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
from scipy import ndimage
from scipy import signal
import cv2


def frequest(im, orientim, windsze, minWaveLength, maxWaveLength):
    rows, cols = np.shape(im)

    # Find mean orientation within the block. This is done by averaging the
    # sines and cosines of the doubled angles before reconstructing the
    # angle again.  This avoids wraparound problems at the origin.

    cosorient = np.mean(np.cos(2 * orientim))
    sinorient = np.mean(np.sin(2 * orientim))
    orient = math.atan2(sinorient, cosorient) / 2

    # Rotate the image block so that the ridges are vertical

    # ROT_mat = cv2.getRotationMatrix2D((cols/2,rows/2),orient/np.pi*180 + 90,1)
    # rotim = cv2.warpAffine(im,ROT_mat,(cols,rows))
    rotim = ndimage.rotate(
        im,
        orient / np.pi * 180 + 90,
        axes=(1, 0),
        reshape=False,
        order=3,
        mode="nearest",
    )

    # Now crop the image so that the rotated image does not contain any
    # invalid regions.  This prevents the projection down the columns
    # from being mucked up.

    cropsze = int(np.fix(rows / np.sqrt(2)))
    offset = int(np.fix((rows - cropsze) / 2))
    rotim = rotim[offset : offset + cropsze][:, offset : offset + cropsze]

    # Sum down the columns to get a projection of the grey values down
    # the ridges.

    proj = np.sum(rotim, axis=0)
    dilation = scipy.ndimage.grey_dilation(proj, windsze, structure=np.ones(windsze))

    temp = np.abs(dilation - proj)

    peak_thresh = 2

    maxpts = (temp < peak_thresh) & (proj > np.mean(proj))
    maxind = np.where(maxpts)

    rows_maxind, cols_maxind = np.shape(maxind)

    # Determine the spatial frequency of the ridges by divinding the
    # distance between the 1st and last peaks by the (No of peaks-1). If no
    # peaks are detected, or the wavelength is outside the allowed bounds,
    # the frequency image is set to 0

    if cols_maxind < 2:
        freqim = np.zeros(im.shape)
    else:
        NoOfPeaks = cols_maxind
        waveLength = (maxind[0][cols_maxind - 1] - maxind[0][0]) / (NoOfPeaks - 1)
        if waveLength >= minWaveLength and waveLength <= maxWaveLength:
            freqim = 1 / np.double(waveLength) * np.ones(im.shape)
        else:
            freqim = np.zeros(im.shape)

    return freqim


def ridge_filter(im, orient, freq, kx, ky):
    angleInc = 3
    im = np.double(im)
    rows, cols = im.shape
    newim = np.zeros((rows, cols))

    freq_1d = np.reshape(freq, (1, rows * cols))
    ind = np.where(freq_1d > 0)

    ind = np.array(ind)
    ind = ind[1, :]

    # Round the array of frequencies to the nearest 0.01 to reduce the
    # number of distinct frequencies we have to deal with.

    non_zero_elems_in_freq = freq_1d[0][ind]
    non_zero_elems_in_freq = np.double(np.round((non_zero_elems_in_freq * 100))) / 100

    unfreq = np.unique(non_zero_elems_in_freq)

    # Generate filters corresponding to these distinct frequencies and
    # orientations in 'angleInc' increments.

    sigmax = 1 / unfreq[0] * kx
    sigmay = 1 / unfreq[0] * ky

    sze = int(np.round(3 * np.max([sigmax, sigmay])))

    x, y = np.meshgrid(
        np.linspace(-sze, sze, (2 * sze + 1)), np.linspace(-sze, sze, (2 * sze + 1))
    )

    reffilter = np.exp(
        -(((np.power(x, 2)) / (sigmax * sigmax) + (np.power(y, 2)) / (sigmay * sigmay)))
    ) * np.cos(2 * np.pi * unfreq[0] * x)
    # this is the original gabor filter

    filt_rows, filt_cols = reffilter.shape

    angleRange = int(180 / angleInc)

    gabor_filter = np.array(np.zeros((angleRange, filt_rows, filt_cols)))

    for o in range(0, angleRange):
        # Generate rotated versions of the filter.  Note orientation
        # image provides orientation *along* the ridges, hence +90
        # degrees, and imrotate requires angles +ve anticlockwise, hence
        # the minus sign.

        rot_filt = scipy.ndimage.rotate(reffilter, -(o * angleInc + 90), reshape=False)
        gabor_filter[o] = rot_filt

    # Find indices of matrix points greater than maxsze from the image
    # boundary

    maxsze = int(sze)

    temp = freq > 0
    validr, validc = np.where(temp)

    temp1 = validr > maxsze
    temp2 = validr < rows - maxsze
    temp3 = validc > maxsze
    temp4 = validc < cols - maxsze

    final_temp = temp1 & temp2 & temp3 & temp4

    finalind = np.where(final_temp)

    # Convert orientation matrix values from radians to an index value
    # that corresponds to round(degrees/angleInc)

    maxorientindex = np.round(180 / angleInc)
    orientindex = np.round(orient / np.pi * 180 / angleInc)

    # do the filtering

    for i in range(0, rows):
        for j in range(0, cols):
            if orientindex[i][j] < 1:
                orientindex[i][j] = orientindex[i][j] + maxorientindex
            if orientindex[i][j] > maxorientindex:
                orientindex[i][j] = orientindex[i][j] - maxorientindex
    finalind_rows, finalind_cols = np.shape(finalind)
    sze = int(sze)
    for k in range(0, finalind_cols):
        r = validr[finalind[0][k]]
        c = validc[finalind[0][k]]

        img_block = im[r - sze : r + sze + 1][:, c - sze : c + sze + 1]

        newim[r][c] = np.sum(img_block * gabor_filter[int(orientindex[r][c]) - 1])

    return newim


def ridge_orient(im, gradientsigma, blocksigma, orientsmoothsigma):
    rows, cols = im.shape
    # Calculate image gradients.
    sze = np.fix(6 * gradientsigma)
    if np.remainder(sze, 2) == 0:
        sze = sze + 1

    gauss = cv2.getGaussianKernel(int(sze), gradientsigma)
    f = gauss * gauss.T

    fy, fx = np.gradient(f)
    # Gradient of Gaussian

    # Gx = ndimage.convolve(np.double(im),fx);
    # Gy = ndimage.convolve(np.double(im),fy);

    Gx = signal.convolve2d(im, fx, mode="same")
    Gy = signal.convolve2d(im, fy, mode="same")

    Gxx = np.power(Gx, 2)
    Gyy = np.power(Gy, 2)
    Gxy = Gx * Gy

    # Now smooth the covariance data to perform a weighted summation of the data.

    sze = np.fix(6 * blocksigma)

    gauss = cv2.getGaussianKernel(int(sze), blocksigma)
    f = gauss * gauss.T

    Gxx = ndimage.convolve(Gxx, f)
    Gyy = ndimage.convolve(Gyy, f)
    Gxy = 2 * ndimage.convolve(Gxy, f)

    # Analytic solution of principal direction
    denom = np.sqrt(np.power(Gxy, 2) + np.power((Gxx - Gyy), 2)) + np.finfo(float).eps

    sin2theta = Gxy / denom
    # Sine and cosine of doubled angles
    cos2theta = (Gxx - Gyy) / denom

    if orientsmoothsigma:
        sze = np.fix(6 * orientsmoothsigma)
        if np.remainder(sze, 2) == 0:
            sze = sze + 1
        gauss = cv2.getGaussianKernel(int(sze), orientsmoothsigma)
        f = gauss * gauss.T
        cos2theta = ndimage.convolve(cos2theta, f)
        # Smoothed sine and cosine of
        sin2theta = ndimage.convolve(sin2theta, f)
        # doubled angles

    orientim = np.pi / 2 + np.arctan2(sin2theta, cos2theta) / 2
    return orientim


# def normalise(img, mean, std):
#     normed = (img - np.mean(img)) / (np.std(img))
#     return normed


def ridge_segment(im, blksze, thresh):
    rows, cols = im.shape

    #     im = cv2.equalizeHist(im)
    #     clahe = cv2.createCLAHE(clipLimit=40, tileGridSize=(120, 120))
    #     im = clahe.apply(im)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    im = cv2.filter2D(im, -1, kernel)
    im = cv2.equalizeHist(im)

    # im = normalise(im, 0, 1);  # normalise to get zero mean and unit standard deviation

    new_rows = int(blksze * np.ceil((float(rows)) / (float(blksze))))
    new_cols = int(blksze * np.ceil((float(cols)) / (float(blksze))))

    padded_img = np.zeros((new_rows, new_cols))
    stddevim = np.zeros((new_rows, new_cols))

    padded_img[0:rows][:, 0:cols] = im

    for i in range(0, new_rows, blksze):
        for j in range(0, new_cols, blksze):
            block = padded_img[i : i + blksze][:, j : j + blksze]

            stddevim[i : i + blksze][:, j : j + blksze] = np.std(block) * np.ones(
                block.shape
            )

    stddevim = stddevim[0:rows][:, 0:cols]

    mask = stddevim > thresh

    mean_val = np.mean(im[mask])

    std_val = np.std(im[mask])

    normim = (im - mean_val) / (std_val)

    return (normim, mask)


def ridge_freq(im, mask, orient, blksze, windsze, minWaveLength, maxWaveLength):
    rows, cols = im.shape
    freq = np.zeros((rows, cols))

    for r in range(0, rows - blksze, blksze):
        for c in range(0, cols - blksze, blksze):
            blkim = im[r : r + blksze][:, c : c + blksze]
            blkor = orient[r : r + blksze][:, c : c + blksze]

            freq[r : r + blksze][:, c : c + blksze] = frequest(
                blkim, blkor, windsze, minWaveLength, maxWaveLength
            )

    freq = freq * mask
    freq_1d = np.reshape(freq, (1, rows * cols))
    ind = np.where(freq_1d > 0)

    ind = np.array(ind)
    ind = ind[1, :]

    non_zero_elems_in_freq = freq_1d[0][ind]

    meanfreq = np.mean(non_zero_elems_in_freq)
    medianfreq = np.median(non_zero_elems_in_freq)
    # does not work properly
    return (freq, meanfreq)


def image_enhance(img):
    blksze = 16
    thresh = 0.1
    normim, mask = ridge_segment(img, blksze, thresh)
    # normalise the image and find a ROI
    # cv2.imwrite(f"step1:normim.jpg", (normim))
    # cv2.imwrite(f"step2:mask.jpg", (mask*255))

    gradientsigma = 1
    blocksigma = 7
    orientsmoothsigma = 7
    orientim = ridge_orient(normim, gradientsigma, blocksigma, orientsmoothsigma)
    # find orientation of every pixel
    # cv2.imwrite(f"step3:orientim.jpg", (orientim*255))

    blksze = 38
    windsze = 5
    minWaveLength = 5
    maxWaveLength = 15
    freq, medfreq = ridge_freq(
        normim, mask, orientim, blksze, windsze, minWaveLength, maxWaveLength
    )
    # find the overall frequency of ridges
    # cv2.imwrite(f"step4:freq.jpg", (freq*255))

    freq = medfreq * mask
    kx = 0.65
    ky = 0.65
    newim = ridge_filter(normim, orientim, freq, kx, ky)
    # create gabor filter and do the actual filtering
    # cv2.imwrite(f"step5:newim.jpg", (newim*255))

    # th, bin_im = cv2.threshold(np.uint8(newim),0,255,cv2.THRESH_BINARY);
    return newim > 1


def processImage(img):
    # from image_enhance import image_enhance
    # img_name = "iphoneCropped.jpeg"
    # print(img_name)
    # img = cv2.imread(img_name, 0)

    # cv2.imwrite(f"step0:input.jpg", (img*255))
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # if len(img.shape) > 2:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # img = img[250:950, 900:1912]

    # fig = plt.figure(figsize=(8, 8))

    # ax1 = fig.add_subplot(221)
    # ax2 = fig.add_subplot(222)
    # ax3 = fig.add_subplot(223)
    # ax4 = fig.add_subplot(224)

    # Hide X and Y axes label marks
    # ax1.xaxis.set_tick_params(labelbottom=False)
    # ax1.yaxis.set_tick_params(labelleft=False)
    # ax1.set_xticks([])
    # ax1.set_yticks([])

    # ax2.xaxis.set_tick_params(labelbottom=False)
    # ax2.yaxis.set_tick_params(labelleft=False)
    # ax2.set_xticks([])
    # ax2.set_yticks([])

    # ax3.xaxis.set_tick_params(labelbottom=False)
    # ax3.yaxis.set_tick_params(labelleft=False)
    # ax3.set_xticks([])
    # ax3.set_yticks([])

    # ax4.xaxis.set_tick_params(labelbottom=False)
    # ax4.yaxis.set_tick_params(labelleft=False)
    # ax4.set_xticks([])
    # ax4.set_yticks([])

    # ax1.title.set_text("Input")

    # ax1.imshow(img, cmap="gray")

    rows, cols = np.shape(img)
    aspect_ratio = np.double(rows) / np.double(cols)

    new_rows = 350
    # randomly selected number
    new_cols = new_rows / aspect_ratio

    img = cv2.resize(img, (int(new_cols), int(new_rows)))

    enhanced_img = image_enhance(img)
    # ax2.title.set_text("non-inverted-non-flipped")
    # ax2.imshow(enhanced_img, cmap="gray")
    # plt.show()

    # cv2.imwrite(f"test_{(img_name).lower()}", (enhanced_img*255))

    out = enhanced_img * 255
    out = out.astype("uint8")
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(out)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    min_size = 1
    # your answer image
    finalImage = np.ones((output.shape)) * 255
    # ax3.title.set_text("final")

    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            finalImage[output == i + 1] = 0

    # ax3.title.set_text("inverted-but-not-flipped")
    # ax3.imshow(finalImage, cmap="gray")
    # cv2.imwrite(f"inverted{(img_name).lower()}", (finalImage))

    cv2.flip(finalImage, 1, finalImage)

    # ax4.title.set_text("inverted-and-flipped")
    # ax4.imshow(finalImage, cmap="gray")

    # cv2.imwrite(f"step6:final.jpg", (finalImage*255))

    # cv2.imwrite(f"modifiedtest_oct5_.jpg", (finalImage))
    return finalImage
