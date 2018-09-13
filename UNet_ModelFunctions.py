import numpy as np
import tensorflow.contrib.keras as keras
from keras.models import Model
from keras.layers.core import Lambda
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose, BatchNormalization, Activation, Dropout, core
from keras.layers import merge, Conv3D, MaxPooling3D, UpSampling3D,AveragePooling3D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, LambdaCallback
from keras.initializers import RandomNormal, glorot_uniform
from LossFunctions import dice_coef_loss, dice_coef
from keras import backend as K
from keras.layers.merge import concatenate, add

def unet_2D(img_rows=None, img_cols=None, channels_in=1, channels_out=1, starting_filter_number=8, 
            kernelsize=(3,3), number_of_pool=5, poolsize=(2,2), filter_rate=2, dropout_rate=0.5, 
            final_activation='sigmoid', 
            loss_function=dice_coef_loss, metric = [dice_coef], learn_rate=1e-2, decay_rate=0):

    layer_conv={}
    #initialize a dictionary of all other layers that are not convolution layers (e.g. input, pooling, deconv).
    layer_others={}

    number_of_layers_half = number_of_pool + 1

    number_of_filters_max = np.round((filter_rate**(number_of_layers_half-1))*starting_filter_number)
   # print('max number of filters in U ' + str(number_of_filters_max))

    #first half of U
    layer_others[0] = Input((img_rows, img_cols, channels_in))
    for layer_number in range(1,number_of_layers_half):
        number_of_filters_current = np.round((filter_rate**(layer_number-1))*starting_filter_number)
        drop_rate_layer = dropout_rate 
        #print(drop_rate_layer)
        layer_conv[layer_number] = Dropout(rate=drop_rate_layer)(BatchNormalization()(Conv2D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same', activation='relu')(layer_others[layer_number-1])))
        layer_conv[layer_number] = (BatchNormalization()(Conv2D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same', activation='relu')(layer_conv[layer_number])))
        layer_others[layer_number] = MaxPooling2D(pool_size=poolsize)(layer_conv[layer_number])

    #center of U
    #print(dropout_rate)
    layer_conv[number_of_layers_half] = Dropout(rate=dropout_rate)(BatchNormalization()(Conv2D(filters=np.round((filter_rate**(number_of_layers_half-1))*starting_filter_number), kernel_size=kernelsize, padding='same', activation='relu')(layer_others[number_of_layers_half-1])))
    layer_conv[number_of_layers_half] = Dropout(rate=dropout_rate)(BatchNormalization()(Conv2D(filters=np.round((filter_rate**(number_of_layers_half-1))*starting_filter_number), kernel_size=kernelsize, padding='same', activation='relu')(layer_conv[number_of_layers_half])))

    #second half of U
    for layer_number in range(number_of_layers_half+1,2*number_of_layers_half):
        number_of_filters_current = np.round((filter_rate**(2*number_of_layers_half-layer_number-1))*starting_filter_number)
        drop_rate_layer = dropout_rate 
        #print(drop_rate_layer)
        layer_others[layer_number]=concatenate([Conv2DTranspose(number_of_filters_current, kernel_size=kernelsize, strides=(2, 2), kernel_initializer='glorot_uniform', padding='same')(layer_conv[layer_number-1]), layer_conv[2*number_of_layers_half-layer_number]],axis=3)
        layer_conv[layer_number] = Dropout(rate=drop_rate_layer)(BatchNormalization()(Conv2D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same', activation='relu')(layer_others[layer_number])))
        layer_conv[layer_number] = (BatchNormalization()(Conv2D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same', activation='relu')(layer_conv[layer_number])))

    layer_conv[2 * number_of_layers_half] = Conv2D(channels_out, kernel_size=kernelsize, padding='same', activation=final_activation)(layer_conv[2 * number_of_layers_half - 1])

    #build and compile U
    model = Model(inputs=[layer_others[0]], outputs=[layer_conv[2 * number_of_layers_half]])
    model.compile(optimizer=Adam(lr=learn_rate,decay=decay_rate), loss=loss_function, metrics=metric)
    return model

def unet_2D_Inception(img_rows=None, img_cols=None, channels_in=1, channels_out=1, starting_filter_number=8, 
            kernelsize1=(5,5), kernelsize2=(3,3), kernelsize3=(1,1), number_of_pool=5, poolsize=(2,2), filter_rate=2, dropout_rate=0.5, 
            final_activation='sigmoid', 
            loss_function=dice_coef_loss, metric = [dice_coef], learn_rate=1e-2, decay_rate=0):

#def build_2d_unet_hierarchically_dense(img_rows=None, img_cols=None, channels_in=1, channels_out=1, starting_filter_number=16, kernelsize=(3,3), number_of_pool=5, poolsize=(2,2), expansion_rate=1, return_rate=1, return_exp=1,  dropout_rate=0.5, number_of_CNN_layers=5, CNN_filter_number=16, CNN_dropout_rate=None, compression=0.5, final_activation='relu', loss_function='mse',learn_rate=1e-3, decay_rate=0):
    initializer=glorot_uniform
    layer_conv={}
    layer_others={}


    tot_num_filters=channels_in

    number_of_layers_half=number_of_pool+1

    number_of_filters_max = np.round((filter_rate**(number_of_layers_half-1))*starting_filter_number)
    #print('max number of filters in U ' + str(number_of_filters_max))
    #print('Dropout Rate:')
    layer_others[0] = Input((img_rows, img_cols, channels_in))
    for layer_number in range(1,number_of_layers_half):
        number_of_filters_current = np.round((filter_rate**(layer_number-1))*starting_filter_number)
        drop_rate_layer = dropout_rate * (np.sqrt((number_of_filters_current/number_of_filters_max)))
        #print(drop_rate_layer)
        if layer_number == 1:
            tot_num_filters+=2*number_of_filters_current
        Incept1 = Dropout(rate=drop_rate_layer)(BatchNormalization()(Conv2D(filters=number_of_filters_current, kernel_size=kernelsize1, padding='same',kernel_initializer=initializer(),activation='relu')(layer_others[layer_number-1])))
        Incept2 = Dropout(rate=drop_rate_layer)(BatchNormalization()(Conv2D(filters=number_of_filters_current, kernel_size=kernelsize2, padding='same',kernel_initializer=initializer(),activation='relu')(layer_others[layer_number-1])))
        Incept3 = Dropout(rate=drop_rate_layer)(BatchNormalization()(Conv2D(filters=number_of_filters_current, kernel_size=kernelsize3, padding='same',kernel_initializer=initializer(),activation='relu')(layer_others[layer_number-1])))
        layer_conv[layer_number] = concatenate([Incept1,Incept2,Incept3], axis=-1)
        layer_others[layer_number] = MaxPooling2D(pool_size=poolsize)(layer_conv[layer_number])

    #print(dropout_rate)
    Incept1 = Dropout(rate=drop_rate_layer)(BatchNormalization()(Conv2D(filters=number_of_filters_current, kernel_size=kernelsize1, padding='same',kernel_initializer=initializer(),activation='relu')(layer_others[number_of_layers_half-1])))
    Incept2 = Dropout(rate=drop_rate_layer)(BatchNormalization()(Conv2D(filters=number_of_filters_current, kernel_size=kernelsize2, padding='same',kernel_initializer=initializer(),activation='relu')(layer_others[number_of_layers_half-1])))
    Incept3 = Dropout(rate=drop_rate_layer)(BatchNormalization()(Conv2D(filters=number_of_filters_current, kernel_size=kernelsize3, padding='same',kernel_initializer=initializer(),activation='relu')(layer_others[number_of_layers_half-1])))
    layer_conv[number_of_layers_half] = concatenate([Incept1,Incept2,Incept3], axis=-1)

    for layer_number in range(number_of_layers_half+1,2*number_of_layers_half):
        number_of_filters_current = np.round((filter_rate**(2*number_of_layers_half-layer_number-1))*starting_filter_number)
        drop_rate_layer = dropout_rate * (np.sqrt(np.sqrt((number_of_filters_current / number_of_filters_max))))
        #print(drop_rate_layer)
        layer_others[layer_number]=concatenate([Conv2DTranspose(number_of_filters_current, kernel_size=kernelsize2, strides=(2, 2), kernel_initializer='glorot_uniform', padding='same')(layer_conv[layer_number-1]), layer_conv[2*number_of_layers_half-layer_number]],axis=3)
        layer_conv[layer_number] = Dropout(rate=drop_rate_layer)(BatchNormalization()(Conv2D(filters=number_of_filters_current, kernel_size=kernelsize2, padding='same', activation='relu')(layer_others[layer_number])))
        layer_conv[layer_number] = Dropout(rate=drop_rate_layer)(BatchNormalization()(Conv2D(filters=number_of_filters_current, kernel_size=kernelsize2, padding='same', activation='relu')(layer_conv[layer_number])))

    layer_conv[2 * number_of_layers_half] = Conv2D(channels_out, kernel_size=kernelsize2, padding='same', activation=final_activation)(layer_conv[2 * number_of_layers_half - 1])

    model = Model(inputs=[layer_others[0]], outputs=[layer_conv[2 * number_of_layers_half]])
    model.compile(optimizer=Adam(lr=learn_rate,decay=decay_rate), loss=loss_function, metrics=metric)

    return model


def unet_2D_ResNeXt(img_rows=None, img_cols=None,cardinality=4, grouped_channels = 4, channels_in=1, channels_out=1, starting_filter_number=32, 
            kernelsize=(3,3), number_of_pool=3, poolsize=(2,2), filter_rate=2, dropout_rate=0.5, 
            final_activation='sigmoid', 
            loss_function=dice_coef_loss, metric = [dice_coef], learn_rate=1e-2, decay_rate=0):
    initializer=glorot_uniform
    layer_conv={}
    layer_others={}


    tot_num_filters=channels_in

    number_of_layers_half=number_of_pool+1

    number_of_filters_max = np.round((filter_rate**(number_of_layers_half-1))*starting_filter_number)
    #print('max number of filters in U ' + str(number_of_filters_max))
   # print('Dropout Rate:')
    layer_others[0] = Input((img_rows, img_cols, channels_in))
    for layer_number in range(1,number_of_layers_half):
        number_of_filters_current = np.round((filter_rate**(layer_number-1))*starting_filter_number)
        drop_rate_layer = dropout_rate
        #print(drop_rate_layer)
        if layer_number == 1:
            tot_num_filters+=2*number_of_filters_current
        group_list = []    
        for c in range(cardinality):
            x = Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels])(layer_others[layer_number-1])
            x = Dropout(rate=drop_rate_layer)(BatchNormalization()(Conv2D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same',kernel_initializer=initializer(),activation='relu')(x)))
            group_list.append(x)
           # print("shape of x", x.shape)  
        group_merge = concatenate(group_list)
        y = BatchNormalization()(group_merge)
        x = add([layer_others[layer_number-1], x])
        layer_conv[layer_number] = Activation('relu')(y)
        #print("shape of y", y.shape)  
        layer_others[layer_number] = MaxPooling2D(pool_size=poolsize)(layer_conv[layer_number])

    #print(dropout_rate)
    layer_conv[number_of_layers_half] = Dropout(rate=drop_rate_layer)(BatchNormalization()(Conv2D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same',kernel_initializer=initializer(),activation='relu')(layer_others[number_of_layers_half-1])))

    for layer_number in range(number_of_layers_half+1,2*number_of_layers_half):
        number_of_filters_current = np.round((filter_rate**(2*number_of_layers_half-layer_number-1))*starting_filter_number)
        drop_rate_layer = dropout_rate 
       # print(drop_rate_layer)
        layer_others[layer_number]=concatenate([Conv2DTranspose(number_of_filters_current, kernel_size=kernelsize, strides=(2, 2), kernel_initializer='glorot_uniform', padding='same')(layer_conv[layer_number-1]), layer_conv[2*number_of_layers_half-layer_number]],axis=3)
        layer_conv[layer_number] = Dropout(rate=drop_rate_layer)(BatchNormalization()(Conv2D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same', activation='relu')(layer_others[layer_number])))
        layer_conv[layer_number] = Dropout(rate=drop_rate_layer)(BatchNormalization()(Conv2D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same', activation='relu')(layer_conv[layer_number])))

    layer_conv[2 * number_of_layers_half] = Conv2D(channels_out, kernel_size=kernelsize, padding='same', activation=final_activation)(layer_conv[2 * number_of_layers_half - 1])

    model = Model(inputs=[layer_others[0]], outputs=[layer_conv[2 * number_of_layers_half]])
    model.compile(optimizer=Adam(lr=learn_rate,decay=decay_rate), loss=loss_function, metrics=metric)

    return model


def unet_2D_ResNeXtMOD(img_rows=None, img_cols=None, channels_in=1, channels_out=1, starting_filter_number=8, 
            kernelsize=(3,3), number_of_pool=5, poolsize=(2,2), filter_rate=2, dropout_rate=0.5, 
            final_activation='sigmoid', 
            loss_function=dice_coef_loss, metric = [dice_coef], learn_rate=1e-2, decay_rate=0):

#def build_2d_unet_hierarchically_dense(img_rows=None, img_cols=None, channels_in=1, channels_out=1, starting_filter_number=16, kernelsize=(3,3), number_of_pool=5, poolsize=(2,2), expansion_rate=1, return_rate=1, return_exp=1,  dropout_rate=0.5, number_of_CNN_layers=5, CNN_filter_number=16, CNN_dropout_rate=None, compression=0.5, final_activation='relu', loss_function='mse',learn_rate=1e-3, decay_rate=0):
    initializer=glorot_uniform
    layer_conv={}
    layer_others={}


    tot_num_filters=channels_in

    number_of_layers_half=number_of_pool+1

    number_of_filters_max = np.round((filter_rate**(number_of_layers_half-1))*starting_filter_number)
   # print('max number of filters in U ' + str(number_of_filters_max))
    #print('Dropout Rate:')
    layer_others[0] = Input((img_rows, img_cols, channels_in))
    for layer_number in range(1,number_of_layers_half):
        number_of_filters_current = np.round((filter_rate**(layer_number-1))*starting_filter_number)
        drop_rate_layer = dropout_rate * (np.sqrt((number_of_filters_current/number_of_filters_max)))
        #print(drop_rate_layer)
        if layer_number == 1:
            tot_num_filters+=2*number_of_filters_current
        R1 = Dropout(rate=drop_rate_layer)(BatchNormalization()(Conv2D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same',kernel_initializer=initializer(),activation='relu')(layer_others[layer_number-1])))
        R2 = Dropout(rate=drop_rate_layer)(BatchNormalization()(Conv2D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same',kernel_initializer=initializer(),activation='relu')(layer_others[layer_number-1])))
        R3 = Dropout(rate=drop_rate_layer)(BatchNormalization()(Conv2D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same',kernel_initializer=initializer(),activation='relu')(layer_others[layer_number-1])))
        layer_conv[layer_number] = concatenate([R1,R2,R3], axis=-1)
        layer_others[layer_number] = MaxPooling2D(pool_size=poolsize)(layer_conv[layer_number])

   # print(dropout_rate)
    R1 = Dropout(rate=drop_rate_layer)(BatchNormalization()(Conv2D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same',kernel_initializer=initializer(),activation='relu')(layer_others[number_of_layers_half-1])))
    R2 = Dropout(rate=drop_rate_layer)(BatchNormalization()(Conv2D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same',kernel_initializer=initializer(),activation='relu')(layer_others[number_of_layers_half-1])))
    R3 = Dropout(rate=drop_rate_layer)(BatchNormalization()(Conv2D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same',kernel_initializer=initializer(),activation='relu')(layer_others[number_of_layers_half-1])))
    layer_conv[number_of_layers_half] = concatenate([R1,R2,R3], axis=-1)

    for layer_number in range(number_of_layers_half+1,2*number_of_layers_half):
        number_of_filters_current = np.round((filter_rate**(2*number_of_layers_half-layer_number-1))*starting_filter_number)
        drop_rate_layer = dropout_rate * (np.sqrt(np.sqrt((number_of_filters_current / number_of_filters_max))))
        #print(drop_rate_layer)
        layer_others[layer_number]=concatenate([Conv2DTranspose(number_of_filters_current, kernel_size=kernelsize, strides=(2, 2), kernel_initializer='glorot_uniform', padding='same')(layer_conv[layer_number-1]), layer_conv[2*number_of_layers_half-layer_number]],axis=3)
        layer_conv[layer_number] = Dropout(rate=drop_rate_layer)(BatchNormalization()(Conv2D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same', activation='relu')(layer_others[layer_number])))
        layer_conv[layer_number] = Dropout(rate=drop_rate_layer)(BatchNormalization()(Conv2D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same', activation='relu')(layer_conv[layer_number])))

    layer_conv[2 * number_of_layers_half] = Conv2D(channels_out, kernel_size=kernelsize, padding='same', activation=final_activation)(layer_conv[2 * number_of_layers_half - 1])

    model = Model(inputs=[layer_others[0]], outputs=[layer_conv[2 * number_of_layers_half]])
    model.compile(optimizer=Adam(lr=learn_rate,decay=decay_rate), loss=loss_function, metrics=metric)

    return model

   
def unet_2D_Symmetry(img_rows=None, img_cols=None, channels_in=1, channels_out=1, starting_filter_number=8, 
            kernelsize=(3,3), number_of_pool=5, poolsize=(2,2), filter_rate=2, dropout_rate=0.5, 
            final_activation='sigmoid', 
            loss_function=dice_coef_loss, metric = [dice_coef], learn_rate=1e-2, decay_rate=0):
    
    layer_conv={}
    #initialize a dictionary of all other layers that are not convolution layers (e.g. input, pooling, deconv).
    layer_others={}

    number_of_layers_half = number_of_pool + 1

    number_of_filters_max = np.round((filter_rate**(number_of_layers_half-1))*starting_filter_number)
    #print('max number of filters in U ' + str(number_of_filters_max))
    r = img_rows
    
    #first half of U
    layer_others[0] = Input((img_rows, img_cols, channels_in))
    for layer_number in range(1,number_of_layers_half):
        number_of_filters_current = np.round((filter_rate**(layer_number-1))*starting_filter_number)
        drop_rate_layer = dropout_rate 
        #print(drop_rate_layer)         
        layer_conv[layer_number] = Dropout(rate=drop_rate_layer)(BatchNormalization()(Conv2D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same', activation='relu')(layer_others[layer_number-1])))
        layer_conv[layer_number] = (BatchNormalization()(Conv2D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same', activation='relu')(layer_conv[layer_number])))
        layer_others[layer_number] = MaxPooling2D(pool_size=poolsize)(layer_conv[layer_number])
        r = int(r) * 0.5
    #center of U
    #print(dropout_rate)
    layer_conv[number_of_layers_half] = Dropout(rate=dropout_rate)(BatchNormalization()(Conv2D(filters=np.round((filter_rate**(number_of_layers_half-1))*starting_filter_number), kernel_size=kernelsize, padding='same', activation='relu')(layer_others[number_of_layers_half-1])))
    layer_conv[number_of_layers_half] = Dropout(rate=dropout_rate)(BatchNormalization()(Conv2D(filters=np.round((filter_rate**(number_of_layers_half-1))*starting_filter_number), kernel_size=kernelsize, padding='same', activation='relu')(layer_conv[number_of_layers_half])))

    #second half of U
    for layer_number in range(number_of_layers_half+1,2*number_of_layers_half):
        number_of_filters_current = np.round((filter_rate**(2*number_of_layers_half-layer_number-1))*starting_filter_number)
        drop_rate_layer = dropout_rate * (np.sqrt(np.sqrt((number_of_filters_current / number_of_filters_max))))
        #print(drop_rate_layer)
 
        xleft = Lambda(lambda z: z[:, 0:r/2, :, :])(layer_conv[layer_number-1])
        xright = Lambda(lambda z: z[:, r/2:r, :, :])(layer_conv[layer_number-1])  
        xleft_rotate = Lambda(lambda z: K.reverse(z,axes=0))(xleft)
        xright_rotate = Lambda(lambda z: K.reverse(z,axes=0))(xright)
        x_leftnew = concatenate(xleft,xright_rotate) 
        x_leftnew_resized = Conv2DTranspose(1, kernel_size=kernelsize, strides=(2, 1), kernel_initializer='glorot_uniform', padding='same')(x_leftnew)
        x_rightnew  = concatenate(xright,xleft_rotate)
        x_rightnew_resized = Conv2DTranspose(1, kernel_size=kernelsize, strides=(2, 1), kernel_initializer='glorot_uniform', padding='same')(x_rightnew)
        
        x_new = concatenate([x_leftnew_resized,x_rightnew_resized])

        layer_others[layer_number]=concatenate([Conv2DTranspose(number_of_filters_current, kernel_size=kernelsize, strides=(2, 2), kernel_initializer='glorot_uniform', padding='same')(x_new), layer_conv[2*number_of_layers_half-layer_number]],axis=3)
        layer_conv[layer_number] = Dropout(rate=drop_rate_layer)(BatchNormalization()(Conv2D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same', activation='relu')(layer_others[layer_number])))
        layer_conv[layer_number] = (BatchNormalization()(Conv2D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same', activation='relu')(layer_conv[layer_number])))
        r = r*2
    layer_conv[2 * number_of_layers_half] = Conv2D(channels_out, kernel_size=kernelsize, padding='same', activation=final_activation)(layer_conv[2 * number_of_layers_half - 1])

    #build and compile U
    model = Model(inputs=[layer_others[0]], outputs=[layer_conv[2 * number_of_layers_half]])
    model.compile(optimizer=Adam(lr=learn_rate,decay=decay_rate), loss=loss_function, metrics=metric)
    return model

def unet_3D(img_rows=None, img_cols=None, img_slc=None, channels_in=1, channels_out=1, starting_filter_number=8, kernelsize=(3,3,3), 
            number_of_pool=5, poolsize=(2,2,2), filter_rate=2, dropout_rate=0.5, final_activation='sigmoid', 
            loss_function=dice_coef_loss, metric = [dice_coef], learn_rate=1e-2, decay_rate=0):

    layer_conv={}
    #initialize a dictionary of all other layers that are not convolution layers (e.g. input, pooling, deconv).
    layer_nonconv={}

    number_of_layers_half = number_of_pool + 1

    number_of_filters_max = np.round((filter_rate**(number_of_layers_half-1))*starting_filter_number)
    #print('max number of filters in U ' + str(number_of_filters_max))

    #first half of U
    layer_nonconv[0] = Input((img_rows, img_cols, img_slc, channels_in))
    for layer_number in range(1,number_of_layers_half):
        number_of_filters_current = np.round((filter_rate**(layer_number-1))*starting_filter_number)
        drop_rate_layer = dropout_rate #* (np.sqrt(np.sqrt((number_of_filters_current/number_of_filters_max))))
        #print(drop_rate_layer)
        #layer_conv[layer_number] = Dropout(rate=drop_rate_layer)(BatchNormalization()(Conv3D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same', activation='relu')(layer_nonconv[layer_number-1])))
        layer_conv[layer_number] = Dropout(rate=drop_rate_layer)(BatchNormalization()(Conv3D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same', activation='relu')(layer_conv[layer_number])))
        layer_nonconv[layer_number] = MaxPooling3D(pool_size=poolsize)(layer_conv[layer_number])

    #center of U
   # print(dropout_rate)
    layer_conv[number_of_layers_half] = Dropout(rate=dropout_rate)(BatchNormalization()(Conv3D(filters=np.round((filter_rate**(number_of_layers_half-1))*starting_filter_number), kernel_size=kernelsize, padding='same', activation='relu')(layer_nonconv[number_of_layers_half-1])))
    layer_conv[number_of_layers_half] = Dropout(rate=dropout_rate)(BatchNormalization()(Conv3D(filters=np.round((filter_rate**(number_of_layers_half-1))*starting_filter_number), kernel_size=kernelsize, padding='same', activation='relu')(layer_conv[number_of_layers_half])))
    layer_conv[number_of_layers_half] = Dropout(rate=dropout_rate)(BatchNormalization()(Conv3D(filters=np.round((filter_rate**(number_of_layers_half-1))*starting_filter_number), kernel_size=kernelsize, padding='same', activation='relu')(layer_conv[number_of_layers_half])))
    layer_conv[number_of_layers_half] = Dropout(rate=dropout_rate)(BatchNormalization()(Conv3D(filters=np.round((filter_rate**(number_of_layers_half-1))*starting_filter_number), kernel_size=kernelsize, padding='same', activation='relu')(layer_conv[number_of_layers_half])))

    #second half of U
    for layer_number in range(number_of_layers_half+1,2*number_of_layers_half):
        number_of_filters_current = np.round((filter_rate**(2*number_of_layers_half-layer_number-1))*starting_filter_number)
        drop_rate_layer = dropout_rate #* (np.sqrt(np.sqrt((number_of_filters_current / number_of_filters_max))))
        #print(drop_rate_layer)
        layer_nonconv[layer_number]=concatenate([Dropout(rate=drop_rate_layer)(BatchNormalization()(Conv3D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same', activation='relu')(UpSampling3D(size=poolsize)(layer_conv[layer_number-1])))), layer_conv[2*number_of_layers_half-layer_number]],axis=-1)
        layer_conv[layer_number] = Dropout(rate=drop_rate_layer)(BatchNormalization()(Conv3D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same', activation='relu')(layer_nonconv[layer_number])))
        layer_conv[layer_number] = Dropout(rate=drop_rate_layer)(BatchNormalization()(Conv3D(filters=number_of_filters_current, kernel_size=kernelsize, padding='same', activation='relu')(layer_conv[layer_number])))

    #Add CNN with output
    layer_conv[2 * number_of_layers_half ] = Conv3D(channels_out, kernel_size=kernelsize, padding='same', activation=final_activation)(layer_conv[2 * number_of_layers_half - 1])

    #build and compile U
    model = Model(inputs=[layer_nonconv[0]], outputs=[layer_conv[2 * number_of_layers_half]])
    model.compile(optimizer=Adam(lr=learn_rate,decay=decay_rate), loss=loss_function, metrics=metric)
    return model