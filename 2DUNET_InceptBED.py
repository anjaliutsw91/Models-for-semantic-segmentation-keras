# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 13:48:39 2018

@author: S426200
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:03:15 2018

@author: S426200
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 13:58:34 2018

@author: S426200
"""


"""
Created on Wed Sep  6 18:53:52 2017

@author: S426200
"""


import os
from skimage.transform import resize
from skimage.io import imsave
from skimage.measure import label, regionprops
from skimage.transform import rotate
from skimage import feature

import numpy as np
import tensorflow.contrib.keras as keras
from keras.models import Model

from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose, BatchNormalization, Activation, Dropout, core
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
#import random
from skimage.io import imread
from PIL import Image
#print (PIL.PILLOW_VERSION)
from keras.callbacks import LearningRateScheduler
from UNet_ModelFunctions import unet_2D

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 160
img_cols = 160
#sl = 48
smooth = 1.

z_dim = []

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def weighted_dice_coef(y_true, y_pred):
    k_width = 24
    w_max = 1.
    edge = K.abs(K.conv3d(y_true, np.float32([[[0,1,0],[1,-4,1],[0,1,0]]]).reshape((3,3,1,1,1)), padding='same', data_format='channels_last'))
    gk = w_max * np.ones((k_width,k_width, 1,1,1), dtype='float32') / 4.
    x_edge = K.clip(K.conv3d(edge, gk, padding='same', data_format='channels_last'), 0., w_max)
    w_f      = K.flatten(x_edge + 1.)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(w_f * y_true_f * y_pred_f)
    return (2. * intersection  + smooth) / (K.sum(w_f * y_true_f) + K.sum(w_f * y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
   
def weighted_dice_loss(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * m1) + K.sum(w * m2) + smooth)
    loss = 1. - K.sum(score)
    return loss

def weighted_bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    averaged_mask = K.pool2d(
            y_true, pool_size=(11, 11), strides=(1, 1), padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss = weighted_dice_loss(y_true, y_pred, weight)
    return loss



def train_and_predict():
    print('Loading and preprocessing train data...')

    Pat = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,55,56,58,59,61,100]

    for p in range(102,323):
        Pat.append(p)
    
    Pat.remove(12)
    Pat.remove(13)
    Pat.remove(104)    
    Pat.remove(105)
    Pat.remove(124)
    Pat.remove(127)
    Pat.remove(134)
    Pat.remove(152)
    Pat.remove(179)
    Pat.remove(187)
    Pat.remove(211)
    Pat.remove(235)
    Pat.remove(239)
    Pat.remove(274)
    Pat.remove(277)
    Pat.remove(278)
    #TEST = Pat[150:176] 
    roi = [4,5,7,8]#,4,5,7]#[2,3,4,5,7]#,8]     
    Pat_temp = []    
    pdata_dir = '/mnt/md0/anjali/Original/'
    for p1 in Pat:
        #print (p1)
        try:
            for i in roi:
                #print (i)
                roi_f =  open(os.path.join(pdata_dir,'ROI_{0}_{1}_{2}.tiff'.format(p1,0,i)))         
        except:
            #print (i,p1, "removed")
            Pat_temp.append(p1)
            continue;
    for a in Pat_temp:
        Pat.remove(a)
#    z_dim = []
#
#    ZDIMarray = np.load('/mnt/md0/anjali/Original/z_dim.npy')
#
#    for z in range(0,len(ZDIMarray)):
#        z_dim.append(ZDIMarray[z])
    print("Calculating z_dim for ", len(Pat))
    z_dim = []
    for i1 in Pat:
        zd = 0;
        t1 = 0;
        while(t1==0):
            ctfile = os.path.join(pdata_dir,'CT_{0}_{1}.tiff'.format(i1,zd)) 
        #print (ct_file)
            try:
                head_f = open(ctfile, 'r', encoding='utf8')
                zd = zd + 1;
            except FileNotFoundError:
            #j = 0;
                t1= 1;
                if zd == 0:
                   print("no patient",i1)
                continue;  
        z_dim.append(zd)  
    np.save('z_dim_BED.npy',z_dim)

    roi = [7]
    Pat_Set1 = Pat[0:20]
    Pat_Set2 = Pat[20:40]
    Pat_Set3 = Pat[40:60]
    Pat_Set4 = Pat[60:80]
    Pat_Set5 = Pat[80:100]    
    Pat_Set6 = Pat[100:120]
    Pat_Set7 = Pat[120:140]
    Pat_Set8 = Pat[140:160]
    Pat_Set9 = Pat[160:180]
    Pat_Set10 = Pat[180:200]  
    
    TRAIN = Pat_Set1 + Pat_Set3 + Pat_Set4 + Pat_Set5 + Pat_Set6 + Pat_Set7 + Pat_Set8+ Pat_Set9+ Pat_Set10 
    VALID = Pat_Set2
 
    zdim_Set1 = z_dim[0:20]
    zdim_Set2 = z_dim[20:40]
    zdim_Set3 = z_dim[40:60]
    zdim_Set4 = z_dim[60:80]
    zdim_Set5 = z_dim[80:100] 
    zdim_Set6 = z_dim[100:120]
    zdim_Set7 = z_dim[120:140]
    zdim_Set8 = z_dim[140:160]
    zdim_Set9 = z_dim[160:180]
    zdim_Set10 = z_dim[180:200] 
    
    z_TRAIN = zdim_Set1 +  zdim_Set3 + zdim_Set4 + zdim_Set5 + zdim_Set6 + zdim_Set7 + zdim_Set8+ zdim_Set9 + zdim_Set10
    z_VALID =  zdim_Set2
    TRAIN_RANGE = len(TRAIN)
    VALID_RANGE = len(VALID)


    preddata_dir = r'/mnt/md0/anjali/prediction7/'
    roi = [7]#,4,5,7]#,8]#[4,5,7,8]#[2,3,4,5,7]#,8]     


    z_starttrain = []#[0]*len(TRAIN)
    z_endtrain = [0]*len(TRAIN)
    
    i = 0
    for x1 in TRAIN:
        #print(x1)
        z_start1check = 0
        for nroi in roi:
            count1 = 0
            for z1 in range(0,z_TRAIN[i]):  
                roi_file = os.path.join(pdata_dir,'ROI_{0}_{1}_{2}.tiff'.format(x1,z1,nroi)) 
                img_roi = imread(roi_file).astype('uint8')
                SUM = np.sum(img_roi)
                if SUM > 0 and count1 == 0:
                   #print(SUM)
                   if nroi == 7:
                      if z_start1check == 0:
                         #print ('start1', z2)
                         z_start1check = 1
                         z_starttrain.append(z1)
                         count1 = 1 
                         #print ('start', x1, z1)
                      else:
                         count1 = 1                            
                         #print ('start other', z2)

                if count1 == 1 and SUM < 1:
                   if nroi == 7:
                      #print ('end',x1, z1, SUM)
                      if z1 > z_endtrain[i]:
                         z_endtrain[i] = z1
                         count1 = 0
       
        i = i+1

    z_startvalid = []
    z_endvalid = [0]*len(VALID)

    i2 = 0
    
    for x2 in VALID:  
        z_start2check = 0
        for nroi in roi:
            count2 = 0
            for z2 in range(0,z_VALID[i2]):  
                roi_file = os.path.join(preddata_dir,'test_result_{0}_{1}_{2}.tiff'.format(x2,z2,nroi)) 
                img_roi = imread(roi_file).astype('uint8')
                SUM = np.sum(img_roi)
                if SUM > 5 and count2 == 0:
                   if nroi == 7:
                      if z_start2check == 0:
                         #print ('valid start1', z2)
                         z_start2check = 1
                         z_startvalid.append(z2)
                         count2 = 1 
                      else:
                         count2 = 1                            
                         #print ('start other', z2)

                if count2 == 1 and SUM < 5:
                   if nroi == 7:
                      #print (z2, SUM)
                      if z2 > z_endvalid[i2]:
                         z_endvalid[i2] = z2
                         count2 = 0
        i2 = i2+1

    #print(len(z_TRAIN),len(z_starttrain), len(z_endtrain))
    z_starttrainact = []
    z_endtrainact = []
    z_startvalidact = []
    z_endvalidact = []
    
    for l in range(0,len(TRAIN)): 
        if(z_starttrain[l] >=5 and z_TRAIN[l] - z_endtrain[l] >= 5):
          z_starttrainact.append(z_starttrain[l] - 5)
          z_endtrainact.append(z_endtrain[l] + 5)
        elif(z_starttrain[l]<5 and z_TRAIN[l]- z_endtrain[l] > 5):
          z_starttrainact.append(0)
          z_endtrainact.append(z_endtrain[l] + 5)
        elif(z_starttrain[l]>5 and z_TRAIN[l] - z_endtrain[l] < 5):
          z_starttrainact.append(z_starttrain[l] - 5)
          z_endtrainact.append(z_TRAIN[l])   
        elif(z_starttrain[l]<5 and z_TRAIN[l] - z_endtrain[l] < 5):
          z_starttrainact.append(0)
          z_endtrainact.append(z_TRAIN[l])   
    
    for l in range(0,len(VALID)): 
        if(z_startvalid[l]  >=5 and z_endvalid[l] >= 5):
          z_startvalidact.append(z_startvalid[l] - 5)
          z_endvalidact.append(z_endvalid[l] + 5)
        elif(z_startvalid[l] <5 and z_VALID[l]- z_endvalid[l] > 5):
          z_startvalidact.append(0)
          z_endvalidact.append(z_endvalid[l] + 5)
        elif(z_startvalid[l] >5 and z_VALID[l] - z_endvalid[l] < 5):
          z_startvalidact.append(z_startvalid[l] - 5)
          z_endvalidact.append(z_VALID[l])   
        elif(z_startvalid[l] <5 and z_VALID[l] - z_endvalid[l] < 5):
          z_startvalidact.append(0)
          z_endvalidact.append(z_VALID[l])   

    sl1 = 0   
    X_centtrain = []
    Y_centtrain = []
    for x1 in TRAIN:
        y_cent = 0
        x_cent = 0
        y0 = 0
        x0 = 0
        count = 0
        a = []
        for z1 in range(z_starttrainact[sl1],z_endtrainact[sl1]):
    
            ROI_file_pro = os.path.join(pdata_dir,'ROI_{0}_{1}_7.tiff'.format(x1,z1))
           
            img_roi = imread(ROI_file_pro)        
            label_img = label(img_roi)
            regions = regionprops(label_img)
            for props in regions:
                x0, y0 = props.centroid
                #print (x0,y0)
                count += 1
                a.append(y0)
                y_cent += y0
                x_cent += x0
        Y = y_cent/count
        X = x_cent/count
        X_centtrain.append(int(X))
        Y_centtrain.append(int(Y))
        sl1 += 1      

    sl2 = 0   
    X_centvalid = []
    Y_centvalid = []
    for x2 in VALID:
        y_cent = 0
        x_cent = 0
        y0 = 0
        x0 = 0
        count = 0
        a = []
        print(x2,z_startvalidact[sl2],z_endvalidact[sl2])
        
        for z2 in range(z_startvalidact[sl2],z_endvalidact[sl2]):
    
            ROI_file_pro = os.path.join(preddata_dir,'test_result_{0}_{1}_7.tiff'.format(x2,z2))
            img_roi = imread(ROI_file_pro)        
            #img_roi = img_roi.resize((512,512), Image.BICUBIC)   
            #img_roi = np.array(img_roi)
            label_img = label(img_roi)
            regions = regionprops(label_img)
            for props in regions:
                x0, y0 = props.centroid
                #print (x0,y0)
                count += 1
                a.append(y0)
                y_cent += y0
                x_cent += x0
        Y = y_cent/count
        X = x_cent/count
        X_centvalid.append(int(X))
        Y_centvalid.append(int(Y))
        sl2 += 1    

    print(len(VALID),len(X_centvalid),len(Y_centvalid))
#    total_slices = 0
#    for ztest in z_dim[0:180]:
#        total_slices += ztest
#    print("total_slices",total_slices)
    train_slices = 0
    #test_slices = 0
    valid_slices = 0

    for s in range(0,TRAIN_RANGE):
        train_slices += (z_endtrainact[s]-z_starttrainact[s])
        #print(TRAIN[s],z_starttrain[s],z_endtrain[s],z_endtrainact[s],z_starttrainact[s],train_slices)
    for s_valid in range(0,VALID_RANGE):
        valid_slices += (z_endvalidact[s_valid]-z_startvalidact[s_valid])
#    for s_test in range(TRAIN_RANGE+VALID_RANGE,len(Pat)):
#        test_slices += z_dim[s_test]
        
    n_train = train_slices
#    n_test = test_slices
    n_valid = valid_slices

    i1 = 0;
    #i2 = 0;
    i3 = 0
    j1 = 0;
    j3 = 0;
    train_imgs_in = np.ndarray((n_train, img_rows, img_cols, 1), dtype=np.uint8)
    train_imgs_roi = np.ndarray((n_train, img_rows, img_cols, 1), dtype=np.uint8)
#    test_imgs_in = np.ndarray((n_test, img_rows, img_cols, nch), dtype=np.uint8)
#    test_imgs_roi = np.ndarray((n_test, img_rows, img_cols, nch), dtype=np.uint8)
    valid_imgs_in = np.ndarray((n_valid, img_rows, img_cols, 1), dtype=np.uint8)
    valid_imgs_roi = np.ndarray((n_valid, img_rows, img_cols, 1), dtype=np.uint8)
    

    for x1 in TRAIN:
        xof1 = int(X_centtrain[j1]-80)
        xof2 = int(X_centtrain[j1]+80)
        yof1 = int(Y_centtrain[j1]-80)
        yof2 = int(Y_centtrain[j1]+80)
        
        for z1 in range(z_starttrainact[j1],z_endtrainact[j1]):
            train_img_in  = imread(os.path.join(pdata_dir,"CT_{}_{}.tiff".format(x1,z1)))
            #train_img_roi = imread(os.path.join(pdata_dir,"ROI_{}_{}_7.tiff".format(x1,z1))) 
            path = os.path.join(pdata_dir,"ROI_{}_{}_7.tiff".format(x1,z1))
            train_img_roi = Image.open(path)            
            image_histogram, bins = np.histogram(train_img_in.flatten(), 255, normed=True)
            cdf = image_histogram.cumsum() # cumulative distribution function
            cdf = 255 * cdf / cdf[-1] # normalize
            image_equalized = np.interp(train_img_in.flatten(), bins[:-1], cdf)
            img_in_he = image_equalized.reshape(train_img_in.shape)
            train_img_in= Image.fromarray(img_in_he)      
            #train_img_roi = Image.fromarray(np.array(train_img_roi)) 
            train_img_in = np.array(train_img_in)
            train_img_roi = np.array(train_img_roi)
            train_imgs_in[i1,:,:,0] =  train_img_in[xof1:xof2,yof1:yof2]
            train_imgs_roi[i1,:,:,0] = train_img_roi[xof1:xof2,yof1:yof2]            
            i1 = i1 +1;
        j1 = j1 + 1;
        
    for x3 in VALID:  
        xof1 = int(X_centvalid[j3]-80)
        xof2 = int(X_centvalid[j3]+80)
        yof1 = int(Y_centvalid[j3]-80)
        yof2 = int(Y_centvalid[j3]+80)
        
        for z3 in range(z_startvalidact[j3],z_endvalidact[j3]):
            valid_img_in  = imread(os.path.join(pdata_dir,"CT_{}_{}.tiff".format(x3,z3)))
            #valid_img_roi = imread(os.path.join(pdata_dir,"ROI_{}_{}_7.tiff".format(x3,z3)))   
            valid_img_roi = Image.open(os.path.join(pdata_dir,"ROI_{}_{}_7.tiff".format(x3,z3)))            
            image_histogram, bins = np.histogram(valid_img_in.flatten(), 255, normed=True)
            cdf = image_histogram.cumsum() # cumulative distribution function
            cdf = 255 * cdf / cdf[-1] # normalize
            image_equalized = np.interp(valid_img_in.flatten(), bins[:-1], cdf)
            img_in_he = image_equalized.reshape(valid_img_in.shape)
            valid_img_in= Image.fromarray(img_in_he)               
            valid_img_in = np.array(valid_img_in)
            valid_img_roi = np.array(valid_img_roi)
            valid_imgs_in[i3,:,:,0] = (valid_img_in[128:384,128:384])[xof1:xof2,yof1:yof2]
            valid_imgs_roi[i3,:,:,0] = (valid_img_roi[128:384,128:384])[xof1:xof2,yof1:yof2]
            i3 = i3 +1;   
        j3 = j3 + 1;  

#    for x2 in TEST:
#        for z2 in range(0,z_dim[j1]):
#            test_img_in  = imread(os.path.join(pdata_dir,"CT_{}_{}.tiff".format(x2,z2)))
#            #test_img_roi = imread(os.path.join(pdata_dir,"ROI_{}_{}_7.tiff".format(x2,z2)))   
#            test_img_roi = Image.open(os.path.join(pdata_dir,"ROI_{}_{}_7.tiff".format(x2,z2)))            
#            image_histogram, bins = np.histogram(test_img_in.flatten(), 255, normed=True)
#            cdf = image_histogram.cumsum() # cumulative distribution function
#            cdf = 255 * cdf / cdf[-1] # normalize
#            image_equalized = np.interp(test_img_in.flatten(), bins[:-1], cdf)
#            img_in_he = image_equalized.reshape(test_img_in.shape)
#            test_img_in= Image.fromarray(img_in_he)               
#            test_img_in = test_img_in.resize((128,128), Image.BICUBIC)
#            test_img_roi = test_img_roi.resize((128,128), Image.BICUBIC)
#            test_img_in = np.array(test_img_in)
#            test_img_roi = np.array(test_img_roi)
#            test_imgs_in[i2,:,:,0] = test_img_in
#            test_imgs_roi[i2,:,:,0] = test_img_roi
#            i2 = i2 +1;   
#        j1 = j1 + 1; 
        
    imgs_train = train_imgs_in.astype('float32')
    mean = np.mean(imgs_train) 
    std = np.std(imgs_train)  
    
    imgs_train -= mean
    imgs_train /= std

    imgs_roi_train = train_imgs_roi.astype('float32')
    imgs_roi_train /= 255.  
    
    print("Mean of train is {0}".format(mean))
    print("STD of train is {0}".format(std))

#    imgs_test = test_imgs_in.astype('float32')
#    meant = np.mean(imgs_test) 
#    stdt = np.std(imgs_test)  
#
#    imgs_test -= meant
#    imgs_test /= stdt
#
#    imgs_roi_test = test_imgs_roi.astype('float32')
#    imgs_roi_test /= 255.  
#    
#    print("Mean of test is {0}".format(meant))
#    print("STD of test is {0}".format(stdt))

    imgs_valid = valid_imgs_in.astype('float32')
    meanv = np.mean(imgs_valid) 
    stdv = np.std(imgs_valid)  

    imgs_valid -= meanv
    imgs_valid /= stdv

    imgs_roi_valid = valid_imgs_roi.astype('float32')
    imgs_roi_valid /= 255.  
    
    print("Mean of valid is {0}".format(meanv))
    print("STD of valid is {0}".format(stdv))

#    print('Creating and compiling model...')

    model = unet_2D(img_rows=None, img_cols=None, channels_in=1, channels_out=1, starting_filter_number=32, 
            kernelsize=(3,3), number_of_pool=5, poolsize=(2,2), filter_rate=2, dropout_rate=0.5, 
            final_activation='sigmoid', 
            loss_function=weighted_bce_dice_loss, metric = [dice_coef], learn_rate=1e-2, decay_rate=0)

    #model.summary()
    #model.load_weights('weightsBED2D.h5')
    model_checkpoint = ModelCheckpoint('weightsBED2D7.h5', monitor='val_loss', save_best_only=True)
    print('Fitting model...')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=35, min_lr=0.0001)
    lr_print_callback = LambdaCallback(on_epoch_begin=lambda epoch,logs: print("Current Learning rate = ",K.get_value(model.optimizer.lr)))
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=45, verbose=0, mode='auto')       
    history = model.fit(imgs_train, imgs_roi_train, batch_size= 10, epochs= 300, verbose=1, shuffle=True,
              validation_split=0,validation_data =[imgs_valid, imgs_roi_valid], 
              callbacks=[model_checkpoint,reduce_lr,lr_print_callback,early_stop])
    
       # list all data in history
    print(history.history.keys())
# summarize history for accuracy
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    #plt.show()
    plt.savefig('dicebed2D7.jpg')
# summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('lossbed2D7.jpg')
    plt.show()

    print('Loading and preprocessing test data...')
    
    print('Loading saved weights...')

    model.load_weights('weightsBED2D7.h5')


    print('Predicting masks on test data...')

    imgs_roi_pred = model.predict(imgs_valid, verbose=1)
    #np.save('imgs_roi_testpred.npy', imgs_roi_testpred)
    return imgs_roi_pred, imgs_roi_valid,imgs_roi_train, VALID,TRAIN, z_startvalidact, z_endvalidact, X_centvalid, Y_centvalid,z_starttrainact,z_endtrainact
    print('Saving predicted masks to files...')


#
if __name__ == '__main__':
    #train_and_predict()
    imgs_roi_pred, imgs_roi_valid,imgs_roi_train, VALID, TRAIN,z_startvalidact, z_endvalidact, X_centvalid, Y_centvalid,z_starttrainact,z_endtrainact = train_and_predict()
    data_out = '/mnt/md0/anjali/prediction7fine/'
    roi = [7]
    np.save('/mnt/md0/anjali/prediction7fine/valid.npy',VALID)
    np.save('/mnt/md0/anjali/prediction7fine/z_startvalid.npy',z_startvalidact)
    np.save('/mnt/md0/anjali/prediction7fine/z_endvalid.npy',z_endvalidact)
    np.save('/mnt/md0/anjali/prediction7fine/X_centvalid.npy',X_centvalid)
    np.save('/mnt/md0/anjali/prediction7fine/Y_centvalid.npy',Y_centvalid)
    #data_out = '/home/anjali/Desktop/ProstateIMRT81/DATA/PREDICTION/'
   
    print("abc")
    k = 0
    j = z_startvalidact[k]
    for i in range(0,len(imgs_roi_pred)):
        Pat_n = VALID[k]
        im = imgs_roi_pred[i,:,:,0]   
        file = os.path.join(data_out+'test_result_{0}_{1}.tiff'.format(Pat_n,j))
        imsave(file, im)
        im1 = imgs_roi_valid[i,:,:,0]  
        file1 = os.path.join(data_out+'test_{0}_{1}.tiff'.format(Pat_n,j))
        imsave(file1, im1)
        if j==z_endvalidact[k]-1:
           print(Pat_n,j)
           if k < len(VALID)-1:
              k = k + 1
              j = z_startvalidact[k]
        else:
            j = j+1;
#    k1 = 0
#    j = z_starttrainact[k1]
#    for i in range(0,len(imgs_roi_train)):
#        print ("train",Pat_n)
#        Pat_n = TRAIN[k1]
#        im1 = imgs_roi_train[i,:,:,0]  
#        file1 = os.path.join(data_out+'train_{0}_{1}.tiff'.format(Pat_n,j))
#        imsave(file1, im1)
#        if j==z_endtrainact[k1]-1:
#           print(Pat_n,j)
#           k1 = k1 + 1
#           j = z_starttrainact[k1]
#        else:
#           j = j+1;
