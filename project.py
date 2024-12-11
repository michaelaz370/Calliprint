import os
import cv2
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from types import NoneType
from datetime import datetime
import statistics
from statistics import mean
import copy
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

# DL
## tensorflow
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, concatenate, Lambda
from keras.optimizers import Adam
from keras.metrics import BinaryAccuracy
###overfitting
from keras.regularizers import l2
##sklearn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


def Prepro2img(path_L, path_R, subimg_numb=None, subimg_size=150, resize_max=600, all=True, limitblob=2):
    """
    :param path_L: path to the left document
    :param path_R: path to the right document
    :param subimg_numb: maximal number of patches
    :param subimg_size: size of the patch
    :param resize_max: maximal size of one of the dimensions height or width
    :param all: preprocess all images or not
    :param limitblob: maximal number of blob to erase
    :return: preprocessed data
    """
    subimgs_L = getSubImg(path_L, subimg_numb=subimg_numb, subimg_size=subimg_size, resize_max=resize_max, all=all,
                          limitblob=limitblob)
    subimgs_R = getSubImg(path_R, subimg_numb=subimg_numb, subimg_size=subimg_size, resize_max=resize_max, all=all,
                          limitblob=limitblob)


    subs = [subimgs_L, subimgs_R]
    paths = [path_L, path_R]
    min_len = min(len(subimgs_L), len(subimgs_R))
    train_x = [list(), list()]
    for i in range(len(subs)):
        for j in range(min_len):
            # if cv2.imwrite(f"{paths[i].split('/')}_{str(i)}", subs[i][j]) != True:
            #     print('img not saved with success')

            ## normalize
            img = subs[i][j]/255
            ## reshape
            img = img.reshape(img.shape[0], img.shape[1], 1)
            ## list
            train_x[i].append(img)

        train_x[i] = np.array(train_x[i])

    return train_x


def getSubImg(original_img_path, subimg_numb=None, subimg_size=150, resize_max=600, all=True, limitblob=2):
    """
    original_img_path: string, path of the image to divide
    subimg_numb: integer, number of sub-images per photo
    subimg_size: integer, size of sub-image
    resize_max:
    all:
    limitblob:

    returns: an array of sub-images in format subimg_size*subimg_size  from an original image
    """
    """
    shortnames:
    r: row
    c: col
    """
    # Read the image
    img = cv2.imread(original_img_path)
    # crop
    croped_img = auto_crop_white_regions(img, all=all, limitblob=limitblob)
    # shrink
    shrinked_img = resizeAndPad(croped_img, subimg_size, resize_max)
    # divide
    subimgs = DivideImg(shrinked_img, subimg_size, subimg_numb)

    return subimgs


def resizeAndPad(img, subimg_size=150, resize_max=600, colorpad=[255, 255, 255]):
    """
    img: np array of pixels
    subimg_size:
    resize_max: int, size of the image
    colorpad: array of bgr numbers

    returns: new image
    """

    height, width = img.shape[:2]
    h_size, w_size = resize_max, resize_max

    if height > h_size or width > w_size:  # shrink
        interp = cv2.INTER_AREA
    else:  # strech
        interp = cv2.INTER_CUBIC

    ratio = width / height

    # print(f'before resize: {img.shape}')
    if ratio > 1:
        new_width = w_size
        new_height = np.round(new_width / ratio).astype(int)

        # pad
        h_size = np.ceil(new_height / subimg_size).astype(int) * subimg_size
        pad = abs(h_size - new_height) / 2
        pad_top, pad_bot = np.floor(pad).astype(int), np.ceil(pad).astype(int)
        pad_left, pad_right = 0, 0

    elif ratio < 1:
        new_height = h_size
        new_width = np.round(new_height * ratio).astype(int)

        # pad
        w_size = np.ceil(new_width / subimg_size).astype(int) * subimg_size
        pad = abs(w_size - new_width) / 2
        pad_left, pad_right = np.floor(pad).astype(int), np.ceil(pad).astype(int)
        pad_top, pad_bot = 0, 0

    else:  # square
        new_height, new_width = h_size, w_size
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    img_scaled = cv2.resize(img, (new_width, new_height), interpolation=interp)

    # padding or cropping
    if new_height < subimg_size or new_width < subimg_size:
        img_scaled = cv2.copyMakeBorder(src=img_scaled, top=pad_top, bottom=pad_bot, left=pad_left, right=pad_right,
                                        borderType=cv2.BORDER_CONSTANT, value=colorpad)
    else:
        extra_h = new_height % subimg_size
        extra_w = new_width % subimg_size
        extra_h_pc = extra_h / new_height
        extra_w_pc = extra_w / new_width
        blank_h_pc = ((subimg_size - extra_h) / 2) / subimg_size if extra_h > 0 else 0
        blank_w_pc = ((subimg_size - extra_w) / 2) / subimg_size if extra_w > 0 else 0

        img_scaled = cv2.copyMakeBorder(src=img_scaled, top=pad_top, bottom=pad_bot, left=pad_left, right=pad_right,
                                        borderType=cv2.BORDER_CONSTANT, value=colorpad)
        # if  (extra_h_pc >= 0.20 and blank_h_pc <= 0.30) or (extra_w_pc >= 0.20 and blank_w_pc <= 0.25): # max 20% blank rows per patches
        #   img_scaled = cv2.copyMakeBorder(src=img_scaled, top=pad_top, bottom=pad_bot, left=pad_left, right=pad_right, borderType=cv2.BORDER_CONSTANT, value=colorpad)
        # else: #throw 0-149 px
        #   img_scaled = img_scaled[:new_height-extra_h, :new_width-extra_w]

    # print(f'after resize: {img_scaled.shape}')
    return img_scaled


def DivideImg(img, subimg_size=150, subimg_numb=20):
    """
    img: np array in format (600, n*200) or inverse
    subimg_size: integer, size of sub-image
    subimg_numb:

    returns: an array of sub-images in format sub_size*sub_size  from an original image
    """

    rows, cols = img.shape[:2]
    n_rows = int(rows / subimg_size)
    n_cols = int(cols / subimg_size)

    subimgs = list()

    for r in range(n_rows):
        for c in range(n_cols):
            subimgs.append(img[r * subimg_size: (r + 1) * subimg_size, c * subimg_size: (c + 1) * subimg_size])

    return np.array(subimgs)


def auto_crop_white_regions(img, all=True, limitblob=2):
    """
    crop only the bottom because other part are more meaningful

    img: gray cv2 np array
    all:
    limitblob:

    returns: np array of cropped image
    """

    # Convert image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Crop rows
    img = Crop(img)

    # Crop cols
    img = np.transpose(img)
    img = Crop(img)
    img = np.transpose(img)

    return img


def Crop(img, all=True, limitblob=2):
    """
    img: np array
    all: boolean, if True so remove all blank rows include interlines
    limitblob: integer, for security
    """

    rows_nb = img.shape[0]
    mask = [True] * rows_nb

    if all:
        for i in reversed(range(rows_nb)):
            if False not in (img[i] == 255):  # blank row
                mask[i] = False
    else:
        percent_erase = 0.05  # if 05% or more of the img is blank, crop it
        percent_ignore = 0.01  # if 01% or less of the img is not blank(into blank zone), crop it
        treshold_erase = round(percent_erase * rows_nb)
        treshold_ignore = round(percent_ignore * rows_nb)

        notblank_counter = 0
        blank_bot_counter = 0  # #(blank) under blom
        blank_top_counter = 0  # #(blank) upper blom
        blom_bot = False  # bottom edge of the blom
        blom_top = False  # top edge of the blom

        for i in reversed(range(rows_nb)):
            if False not in (img[i] == 255):  # blank row
                if blom_top:  # it is blank over the blom/line
                    blank_top_counter += 1
                else:
                    if blom_bot:  # it is blank inside the blom/line
                        blom_top = True
                        blank_top_counter += 1
                    else:  # it is blank under the blom/line
                        blank_bot_counter += 1
                        mask[i] = False
            else:  # not blank row: blom or line
                if blom_top:  # we have passed the blom so we have encounter a new blom/line
                    if ((blank_top_counter + blank_bot_counter) >= treshold_erase) and (notblank_counter <= treshold_ignore):
                        for j in reversed(range(i, rows_nb)):
                            mask[j] = False
                    if limitblob <= 0:
                        break
                    else:
                        limitblob -= 1
                        blom_top = False
                        blom_bot = True
                        notblank_counter = 1
                        blank_bot_counter = 0
                        blank_top_counter = 0
                else:
                    if blom_bot:  # we are in the blom/line
                        notblank_counter += 1
                    else:  # we were in blank zone and now we have meet a not blank zone
                        blom_bot = True
                        notblank_counter += 1
    return img[mask]


def TranslatePredict(preds):
    """
    preds: predictions
    """
    pred_2bit = list()
    pred_1bit = list()
    for p in preds:
        if p[0] >= p[1]:
            p_2 = [1.0, 0.0]
            p_1 = 0
        else:
            p_2 = [0.0, 1.0]
            p_1 = 1
        pred_2bit.append(p_2)
        pred_1bit.append(p_1)

    return pred_2bit, pred_1bit


def GetModel(path):
    """
    :param path: string path to the  model
    :return:
    """
    global model
    model = keras.models.load_model(path, compile=False)
    opt = Adam(learning_rate=0.00001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


def InterpretPredict(pred):
    """

    :param pred:
    :return:
    """
    return mean(pred) > 0.5


def GetPredict(path_L, path_R):
    """

    :param path_L:
    :param path_R:
    :return:
    """
    test_x = Prepro2img(path_L, path_R)

    pred = model.predict(test_x)
    pred_2bit, pred_1bit = TranslatePredict(pred)
    interpret = InterpretPredict(pred_1bit)
    score = max(pred[0])
    print('score: ', score)

    return interpret, score


# path_L = r'C:\Users\adam3\OneDrive\DocumentsMine\Principal\ArT\finalproject\MyProject\project\shared\shared_1\images\HHD\img\w3_F_1_form9.tif'
# path_R = r'C:\Users\adam3\OneDrive\DocumentsMine\Principal\ArT\finalproject\MyProject\project\shared\shared_1\images\HHD\img\w2_M_3_form38.tif'
# path2 = r'C:\Users\adam3\OneDrive\DocumentsMine\Principal\ArT\finalproject\MyProject\project\shared\shared_1\images\HHD\img\w3_F_1_form12.tif'

# GetModel('./model.h5')
# print(GetPredict(path_L, path_R))
# GetPredict(path_L, path_R)
