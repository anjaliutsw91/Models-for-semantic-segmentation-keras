import os
from skimage.transform import resize
from skimage.io import imsave
from skimage.measure import label, regionprops
from skimage.transform import rotate
from skimage import feature

import numpy as np
import tensorflow.contrib.keras as keras
from keras.models import Model

from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose, \
    BatchNormalization, Activation, Dropout, core
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LambdaCallback, EarlyStopping
from keras import backend as K

from keras.regularizers import l2
from keras import backend as K
import h5py
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import tensorflow as tf
from numpy import random
# import random
from skimage.io import imread
from PIL import Image
# print (PIL.PILLOW_VERSION)
from keras.callbacks import LearningRateScheduler
from UNet_ModelFunctions import unet_2D

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 160
img_cols = 160
# sl = 48
smooth = 1.

z_dim = []
pdata_dir = '/mnt/md0/anjali/Original/'
pred_dir = '/mnt/md0/anjali/prediction/'


# Function to remove patients from the list that don't have all contours
# Input  : contour list, Patient list
# Output : new patient list
def clean_patients(roi_list, pat):
    pat_temp = []
    for p1 in pat:
        try:
            for i in roi_list:
                roi_f = open(os.path.join(pdata_dir, 'ROI_{0}_{1}_{2}.tiff'.format(p1, 0, i)))
        except:
            pat_temp.append(p1)
            continue;
    for a in pat_temp:
        pat.remove(a)
    return pat

# Function to create a list for patients
# Input  : contour list, list of patients to be removed, contours list
# Output : new patient list
def create_pat_list(roi_list,
                    remove_list=[12, 13, 104, 105, 124, 127, 134, 152, 179, 187, 211, 235, 239, 274, 277, 278]):
    pat = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
           30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 58, 59, 61,
           100]
    for p in range(102, 323):
        pat.append(p)
    for i in remove_list:
        pat.remove(i)
    pat = clean_patients(roi_list, pat)
    return pat

# Function to calculate the number of CT slices for each patients in the patient list
# Input  : Patient list
# Output : list of z values
def z_calc(pat):
    for i1 in pat:
        zd = 0;
        t1 = 0;
        while (t1 == 0):
            ctfile = os.path.join(pdata_dir, 'CT_{0}_{1}.tiff'.format(i1, zd))
            try:
                open(ctfile, 'r', encoding='utf8')
                zd = zd + 1;
            except FileNotFoundError:
                t1 = 1;
                if zd == 0:
                    print("no patient", i1)
                continue;
        z_dim.append(zd)
    np.save('/mnt/md0/anjali/Original/z_dim_BED.npy', z_dim)
    return z_dim

# Function to split data into training and cross validation sets
# Input  : Patient list, z list and number of cross validations to be performed
# Output : training patient and z list, validation patient and z list
def data_split_trainvalid(pat, z_dim, n=10):
    P = len(pat)
    Pat_Set1 = pat[0:int(P / 10)]
    Pat_Set2 = pat[int(P / 10):int((2 * P) / 10)]
    Pat_Set3 = pat[int((2 * P) / 10):int((3 * P) / 10)]
    Pat_Set4 = pat[int((3 * P) / 10):int((4 * P) / 10)]
    Pat_Set5 = pat[int((4 * P) / 10):int((5 * P) / 10)]
    Pat_Set6 = pat[int((5 * P) / 10):int((6 * P) / 10)]
    Pat_Set7 = pat[int((6 * P) / 10):int((7 * P) / 10)]
    Pat_Set8 = pat[int((7 * P) / 10):int((8 * P) / 10)]
    Pat_Set9 = pat[int((8 * P) / 10):int((9 * P) / 10)]
    Pat_Set10 = pat[int((9 * P) / 10):int((10 * P) / 10)]

    TRAIN = Pat_Set1 + Pat_Set3 + Pat_Set4 + Pat_Set5 + Pat_Set6 + Pat_Set7 + Pat_Set8 + Pat_Set9 + Pat_Set10
    VALID = Pat_Set2

    zdim_Set1 = z_dim[0:int(P / 10)]
    zdim_Set2 = z_dim[int(P / 10):int((2 * P) / 10)]
    zdim_Set3 = z_dim[int((2 * P) / 10):int((3 * P) / 10)]
    zdim_Set4 = z_dim[int((3 * P) / 10):int((4 * P) / 10)]
    zdim_Set5 = z_dim[int((4 * P) / 10):int((5 * P) / 10)]
    zdim_Set6 = z_dim[int((5 * P) / 10):int((6 * P) / 10)]
    zdim_Set7 = z_dim[int((6 * P) / 10):int((7 * P) / 10)]
    zdim_Set8 = z_dim[int((7 * P) / 10):int((8 * P) / 10)]
    zdim_Set9 = z_dim[int((8 * P) / 10):int((9 * P) / 10)]
    zdim_Set10 = z_dim[int((9 * P) / 10):int((10 * P) / 10)]

    z_TRAIN = zdim_Set1 + zdim_Set3 + zdim_Set4 + zdim_Set5 + zdim_Set6 + zdim_Set7 + zdim_Set8 + zdim_Set9 + zdim_Set10
    z_VALID = zdim_Set2

    return TRAIN, z_TRAIN, VALID, z_VALID

# Function to split data into training and cross validation sets
# Input  : Patient list, z list and number of cross validations to be performed
# Output : training patient and z list, validation patient and z list
def calculating_slicesofinterest(TRAIN, z_TRAIN, VALID, z_VALID):
    z_starttrain = []  # [0]*len(TRAIN)
    z_endtrain = [0] * len(TRAIN)

    i = 0
    for x1 in TRAIN:
        # print(x1)
        z_start1check = 0
        for nroi in roi:
            count1 = 0
            for z1 in range(0, z_TRAIN[i]):
                roi_file = os.path.join(pdata_dir, 'ROI_{0}_{1}_{2}.tiff'.format(x1, z1, nroi))
                img_roi = imread(roi_file).astype('uint8')
                SUM = np.sum(img_roi)
                if SUM > 0 and count1 == 0:
                    # print(SUM)
                    if nroi == 7:
                        if z_start1check == 0:
                            # print ('start1', z2)
                            z_start1check = 1
                            z_starttrain.append(z1)
                            count1 = 1
                            # print ('start', x1, z1)
                        else:
                            count1 = 1
                            # print ('start other', z2)

                if count1 == 1 and SUM < 1:
                    if nroi == 7:
                        # print ('end',x1, z1, SUM)
                        if z1 > z_endtrain[i]:
                            z_endtrain[i] = z1
                            count1 = 0

        i = i + 1

    z_startvalid = []
    z_endvalid = [0] * len(VALID)

    i2 = 0

    for x2 in VALID:
        z_start2check = 0
        for nroi in roi:
            count2 = 0
            for z2 in range(0, z_VALID[i2]):
                roi_file = os.path.join(preddata_dir, 'test_result_{0}_{1}_{2}.tiff'.format(x2, z2, nroi))
                img_roi = imread(roi_file).astype('uint8')
                SUM = np.sum(img_roi)
                if SUM > 5 and count2 == 0:
                    if nroi == 7:
                        if z_start2check == 0:
                            # print ('valid start1', z2)
                            z_start2check = 1
                            z_startvalid.append(z2)
                            count2 = 1
                        else:
                            count2 = 1
                            # print ('start other', z2)

                if count2 == 1 and SUM < 5:
                    if nroi == 7:
                        # print (z2, SUM)
                        if z2 > z_endvalid[i2]:
                            z_endvalid[i2] = z2
                            count2 = 0
        i2 = i2 + 1
    return z_starttrain, z_endtrain, z_startvalid, z_endvalid

# Function to calculate slices that have masks and add few slices to incorportr error in localization
# Input  : TRAIN, z_starttrain, z_endtrain, z_TRAIN, VALID, z_startvalid, z_endvalid, z_VALID, buff_size
# Output : training patient and z list, validation patient and z list
def calculating_slicesofinterestwithbuffers(TRAIN, z_starttrain, z_endtrain, z_TRAIN, VALID, z_startvalid, z_endvalid,
                                            z_VALID, buff_size=5):
    z_starttrainact = []
    z_endtrainact = []
    z_startvalidact = []
    z_endvalidact = []

    for l in range(0, len(TRAIN)):
        if (z_starttrain[l] >= buff_size and z_TRAIN[l] - z_endtrain[l] >= buff_size):
            z_starttrainact.append(z_starttrain[l] - buff_size)
            z_endtrainact.append(z_endtrain[l] + buff_size)
        elif (z_starttrain[l] < buff_size and z_TRAIN[l] - z_endtrain[l] > buff_size):
            z_starttrainact.append(0)
            z_endtrainact.append(z_endtrain[l] + buff_size)
        elif (z_starttrain[l] > buff_size and z_TRAIN[l] - z_endtrain[l] < buff_size):
            z_starttrainact.append(z_starttrain[l] - buff_size)
            z_endtrainact.append(z_TRAIN[l])
        elif (z_starttrain[l] < buff_size and z_TRAIN[l] - z_endtrain[l] < buff_size):
            z_starttrainact.append(0)
            z_endtrainact.append(z_TRAIN[l])

    for l in range(0, len(VALID)):
        if (z_startvalid[l] >= buff_size and z_endvalid[l] >= buff_size):
            z_startvalidact.append(z_startvalid[l] - buff_size)
            z_endvalidact.append(z_endvalid[l] + buff_size)
        elif (z_startvalid[l] < buff_size and z_VALID[l] - z_endvalid[l] > buff_size):
            z_startvalidact.append(0)
            z_endvalidact.append(z_endvalid[l] + buff_size)
        elif (z_startvalid[l] > buff_size and z_VALID[l] - z_endvalid[l] < buff_size):
            z_startvalidact.append(z_startvalid[l] - buff_size)
            z_endvalidact.append(z_VALID[l])
        elif (z_startvalid[l] < buff_size and z_VALID[l] - z_endvalid[l] < buff_size):
            z_startvalidact.append(0)
            z_endvalidact.append(z_VALID[l])
    return z_starttrainact, z_endtrainact, z_startvalidact, z_endvalidact


def calculate_centroid(train_patlist, z_starttrainact, z_endtrainact, valid_patlist, z_startvalidact, z_endvalidact):
    sl1 = 0
    x_centtrain = []
    y_centtrain = []
    for x1 in train_patlist:
        y_cent = 0
        x_cent = 0
        count = 0
        a = []
        for z1 in range(z_starttrainact[sl1], z_endtrainact[sl1]):
            roi_file_pro = os.path.join(pred_dir, 'ROI_{0}_{1}_7.tiff'.format(x1, z1))
            img_roi = imread(roi_file_pro)
            label_img = label(img_roi)
            regions = regionprops(label_img)
            for props in regions:
                x0, y0 = props.centroid
                count += 1
                a.append(y0)
                y_cent += y0
                x_cent += x0
        y = y_cent / count
        x = x_cent / count
        x_centtrain.append(int(x))
        y_centtrain.append(int(y))
        sl1 += 1

    sl2 = 0
    x_centvalid = []
    y_centvalid = []
    for x2 in valid_patlist:
        y_cent = 0
        x_cent = 0
        count = 0
        a = []
        print(x2, z_startvalidact[sl2], z_endvalidact[sl2])
        for z2 in range(z_startvalidact[sl2], z_endvalidact[sl2]):
            roi_file_pro = os.path.join(pred_dir, 'test_result_{0}_{1}_7.tiff'.format(x2, z2))
            img_roi = imread(roi_file_pro)
            label_img = label(img_roi)
            regions = regionprops(label_img)
            for props in regions:
                x0, y0 = props.centroid
                # print (x0,y0)
                count += 1
                a.append(y0)
                y_cent += y0
                x_cent += x0
        y = y_cent / count
        x = x_cent / count
        x_centvalid.append(int(x))
        y_centvalid.append(int(y))
        sl2 += 1
    return x_centtrain, y_centtrain, x_centvalid, y_centvalid


def create_trainvalid_tensors(train_patlist, z_starttrainact, z_endtrainact, X_centtrain, Y_centtrain,
                              valid_patlist, z_startvalidact, z_endvalidact, X_centvalid, Y_centvalid,
                              cropdim_x=80, cropdim_y=80, roi_interest=7):
    train_slices = 0
    valid_slices = 0
    for s in range(0, len(train_patlist)):
        train_slices += (z_endtrainact[s] - z_starttrainact[s])
    for s_valid in range(0, len(valid_patlist)):
        valid_slices += (z_endvalidact[s_valid] - z_startvalidact[s_valid])

    n_train = train_slices
    n_valid = valid_slices

    i1 = 0
    i3 = 0
    j1 = 0
    j3 = 0
    train_imgs_in = np.ndarray((n_train, img_rows, img_cols, 1), dtype=np.uint8)
    train_imgs_roi = np.ndarray((n_train, img_rows, img_cols, 1), dtype=np.uint8)
    valid_imgs_in = np.ndarray((n_valid, img_rows, img_cols, 1), dtype=np.uint8)
    valid_imgs_roi = np.ndarray((n_valid, img_rows, img_cols, 1), dtype=np.uint8)

    for x1 in train_patlist:
        xof1 = int(X_centtrain[j1] - cropdim_x)
        xof2 = int(X_centtrain[j1] + cropdim_x)
        yof1 = int(Y_centtrain[j1] - cropdim_y)
        yof2 = int(Y_centtrain[j1] + cropdim_y)

        for z1 in range(z_starttrainact[j1], z_endtrainact[j1]):
            train_img_in = imread(os.path.join(pdata_dir, "CT_{}_{}.tiff".format(x1, z1)))
            # train_img_roi = imread(os.path.join(pdata_dir,"ROI_{}_{}_7.tiff".format(x1,z1)))
            path = os.path.join(pdata_dir, "ROI_{}_{}_{}.tiff".format(x1, z1, roi_interest))
            train_img_roi = Image.open(path)
            image_histogram, bins = np.histogram(train_img_in.flatten(), 255, normed=True)
            cdf = image_histogram.cumsum()  # cumulative distribution function
            cdf = 255 * cdf / cdf[-1]  # normalize
            image_equalized = np.interp(train_img_in.flatten(), bins[:-1], cdf)
            img_in_he = image_equalized.reshape(train_img_in.shape)
            train_img_in = Image.fromarray(img_in_he)
            # train_img_roi = Image.fromarray(np.array(train_img_roi))
            train_img_in = np.array(train_img_in)
            train_img_roi = np.array(train_img_roi)
            train_imgs_in[i1, :, :, 0] = train_img_in[xof1:xof2, yof1:yof2]
            train_imgs_roi[i1, :, :, 0] = train_img_roi[xof1:xof2, yof1:yof2]
            i1 = i1 + 1
        j1 = j1 + 1

    for x3 in valid_patlist:
        xof1 = int(X_centvalid[j3] - cropdim_x)
        xof2 = int(X_centvalid[j3] + cropdim_x)
        yof1 = int(Y_centvalid[j3] - cropdim_y)
        yof2 = int(Y_centvalid[j3] + cropdim_y)

        for z3 in range(z_startvalidact[j3], z_endvalidact[j3]):
            valid_img_in = imread(os.path.join(pdata_dir, "CT_{}_{}.tiff".format(x3, z3)))
            # valid_img_roi = imread(os.path.join(pdata_dir,"ROI_{}_{}_7.tiff".format(x3,z3)))
            valid_img_roi = Image.open(os.path.join(pdata_dir, "ROI_{}_{}_{}.tiff".format(x3, z3, roi_interest)))
            image_histogram, bins = np.histogram(valid_img_in.flatten(), 255, normed=True)
            cdf = image_histogram.cumsum()  # cumulative distribution function
            cdf = 255 * cdf / cdf[-1]  # normalize
            image_equalized = np.interp(valid_img_in.flatten(), bins[:-1], cdf)
            img_in_he = image_equalized.reshape(valid_img_in.shape)
            valid_img_in = Image.fromarray(img_in_he)
            valid_img_in = np.array(valid_img_in)
            valid_img_roi = np.array(valid_img_roi)
            valid_imgs_in[i3, :, :, 0] = (valid_img_in[128:384, 128:384])[xof1:xof2, yof1:yof2]
            valid_imgs_roi[i3, :, :, 0] = (valid_img_roi[128:384, 128:384])[xof1:xof2, yof1:yof2]
            i3 = i3 + 1
        j3 = j3 + 1
    return train_imgs_in, train_imgs_roi, valid_imgs_in, valid_imgs_roi


def normalise_data(train_imgs_in, train_imgs_roi, valid_imgs_in, valid_imgs_roi):
    imgs_train = train_imgs_in.astype('float32')
    mean = np.mean(imgs_train)
    std = np.std(imgs_train)
    imgs_train -= mean
    imgs_train /= std
    imgs_roi_train = train_imgs_roi.astype('float32')
    imgs_roi_train /= 255.
    print("Mean of train is {0}".format(mean))
    print("STD of train is {0}".format(std))
    imgs_valid = valid_imgs_in.astype('float32')
    meanv = np.mean(imgs_valid)
    stdv = np.std(imgs_valid)
    imgs_valid -= meanv
    imgs_valid /= stdv
    imgs_roi_valid = valid_imgs_roi.astype('float32')
    imgs_roi_valid /= 255.
    print("Mean of valid is {0}".format(meanv))
    print("STD of valid is {0}".format(stdv))
    return imgs_train, imgs_roi_train, imgs_valid, imgs_roi_valid


def trainvalid_data(roi_list, roi_interest, cropdim_x, cropdim_y, buff_size, n_cv):
    print('Loading and preprocessing train data...')

    pat = create_pat_list(roi_list)

    print("Calculating z_dim for ", len(pat))
    z_dim = z_calc(pat)
    # z_dim = np.load('/mnt/md0/anjali/Original/z_dim_BED.npy').tolist()

    train_patlist, z_train, valid_patlist, z_valid = data_split_trainvalid(pat, z_dim, n_cv)

    z_starttrain, z_endtrain, z_startvalid, z_endvalid = calculating_slicesofinterest(train_patlist, z_train,
                                                                                      valid_patlist, z_valid)
    z_starttrainact, z_endtrainact, z_startvalidact, z_endvalidact = calculating_slicesofinterestwithbuffers(
        train_patlist, z_starttrain, z_endtrain, z_train,
        valid_patlist, z_startvalid, z_endvalid, z_valid,
        buff_size)

    x_centtrain, y_centtrain, x_centvalid, y_centvalid = calculate_centroid(train_patlist, z_starttrainact,
                                                                            z_endtrainact, valid_patlist,
                                                                            z_startvalidact, z_endvalidact)

    train_imgs_in, train_imgs_roi, valid_imgs_in, valid_imgs_roi = create_trainvalid_tensors(train_patlist,
                                                                                             z_starttrainact,
                                                                                             z_endtrainact,
                                                                                             x_centtrain, y_centtrain,
                                                                                             valid_patlist,
                                                                                             z_startvalidact,
                                                                                             z_endvalidact,
                                                                                             x_centvalid, y_centvalid,
                                                                                             cropdim_x, cropdim_y,
                                                                                             roi_interest)
    imgs_train, imgs_roi_train, imgs_valid, imgs_roi_valid = normalise_data(train_imgs_in, train_imgs_roi,
                                                                            valid_imgs_in, valid_imgs_roi)

    return imgs_train, imgs_roi_train, imgs_valid, imgs_roi_valid
