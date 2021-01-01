import os
import re
import sys
import json
import shutil
import pickle
import yaml
import string
import logging
import random
import csv
import itertools

import numpy as np

from tensorflow.keras import utils
from tensorflow.keras.layers import Layer, Input, Dense, Concatenate, BatchNormalization, LeakyReLU, Lambda, Dropout
from tensorflow.keras.losses import binary_crossentropy, mean_squared_error
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam, SGD
from tensorflow.keras.activations import relu, elu, selu, softmax, tanh
from tensorflow.keras.models import Model, model_from_json, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.regularizers import l1,l2
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # removes annoying warning

from talos import Scan, Reporting, Predict, Evaluate, Deploy, Restore, Autom8
from talos.utils.best_model import *
from talos.model.layers import *
from talos.model.normalizers import lr_normalizer
from talos.utils.gpu_utils import parallel_gpu_jobs
import talos

from lbn import LBN, LBNLayer

import matplotlib.pyplot as plt

# Personal files #
import parameters
from preprocessing import PreprocessLayer
from data_generator import DataGenerator
import OneHot 

#################################################################################################
# LossHistory #
#################################################################################################
class LossHistory(tf.keras.callbacks.Callback):
    """ Records the history of the training per epoch and per batch """
    def on_train_begin(self, logs={}):
        self.batch_loss         = {'batch':[], 'loss':[]}
        self.batch_acc          = {'batch':[], 'acc':[]}
        self.epoch_loss         = {'epoch':[], 'loss':[]}
        self.epoch_acc          = {'epoch':[], 'acc':[]}
        self.epoch_val_loss     = {'epoch':[], 'loss':[]}
        self.epoch_val_acc      = {'epoch':[], 'acc':[]}
        self.epoch_lr           = {'epoch':[], 'lr':[]}
        self.epoch_counter      = 0
        self.batch_counter      = 0
        self.epoch_to_batch     = 0

    def on_batch_end(self, batch, logs={}):
        # X value #
        self.batch_loss['batch'].append(batch + self.epoch_to_batch)
        self.batch_acc['batch'].append(batch + self.epoch_to_batch)
        # Y value #
        self.batch_loss['loss'].append(logs.get('loss'))
        self.batch_acc['acc'].append(logs.get('acc'))
        self.batch_counter += 1

    def on_epoch_end(self, epoch, logs={}):
        # X value #
        self.epoch_loss['epoch'].append(epoch)
        self.epoch_acc['epoch'].append(epoch)
        self.epoch_val_loss['epoch'].append(epoch)
        self.epoch_val_acc['epoch'].append(epoch)
        self.epoch_lr['epoch'].append(epoch)
        # Y value #
        self.epoch_loss['loss'].append(logs.get('loss'))
        self.epoch_acc['acc'].append(logs.get('acc'))
        self.epoch_val_loss['loss'].append(logs.get('val_loss'))
        self.epoch_val_acc['acc'].append(logs.get('val_acc'))
        self.epoch_lr['lr'].append(tf.keras.backend.eval(self.model.optimizer.lr))

        # Batch counting #
        self.epoch_counter += 1
        self.epoch_to_batch += self.batch_counter
        self.batch_counter = 0

#################################################################################################
# PlotHistory #
#################################################################################################
def PlotHistory(history):
    """ Takes history from Keras training and makes loss plots (batch and epoch) and learning rate plots """
    #----- Figure -----#
    fig = plt.figure(figsize=(6,9))
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313)
    plt.subplots_adjust(hspace=0.4)

    #----- Plots -----#
    # Per epoch #
    line1 = ax1.plot(history.epoch_loss['epoch'],history.epoch_loss['loss'],'r',label='Loss train')
    line2 = ax1.plot(history.epoch_val_loss['epoch'],history.epoch_val_loss['loss'],'g',label='Loss test')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.set_title('Loss over epochs')

    ax1_2 = ax1.twinx()
    line3 = ax1_2.plot(history.epoch_acc['epoch'],history.epoch_acc['acc'],'r--',label='Accuracy train')
    line4 = ax1_2.plot(history.epoch_val_acc['epoch'],history.epoch_val_acc['acc'],'g--',label='Accuracy test')
    ax1_2.set_ylabel("Accuracy")
    ax1_2.set_ylim(0,1)

    lines = line1+line2+line3+line4
    labels = [l.get_label() for l in lines]
    ax1_2.legend(lines, labels, loc='center right')

    # Per batch #
    line1 = ax2.plot(history.batch_loss['batch'],history.batch_loss['loss'],'b',label='Loss train')
    ax2.set_xlabel('batch')
    ax2.set_ylabel('loss')
    ax2.set_title('Loss over batches')
    #ax2.set_yscale("log")

    ax2_2 = ax2.twinx()
    line2 = ax2_2.plot(history.batch_acc['batch'],history.batch_acc['acc'],'c',label='Accuracy train')
    ax2_2.set_ylabel("Accuracy")
    ax2_2.set_ylim(0,1)

    lines = line1+line2
    labels = [l.get_label() for l in lines]
    ax2_2.legend(lines, labels, loc='center right')

    # LR #
    ax3.plot(history.epoch_lr['epoch'],history.epoch_lr['lr'])
    ax3.set_xlabel('epoch')
    ax3.set_ylabel('LR')
    ax3.set_title('Learning rate over epochs')

    # Save #
    rand_hash = ''.join(random.choice(string.ascii_uppercase) for _ in range(10)) # avoids overwritting
    png_name = 'Loss_%s.png'%rand_hash
    fig.savefig(png_name)
    logging.info('Curves saved as %s'%png_name)

#################################################################################################
# NeuralNetModel#
#################################################################################################
def NeuralNetModel(x_train,y_train,x_val,y_val,params):
    """
    Keras model for the Neural Network, used to scan the hyperparameter space by Talos
    Uses the data provided as inputs
    """
    # Split y = [target,weight], Talos does not leave room for the weight so had to be included in one of the arrays
    w_train = y_train[:,-1]
    w_val = y_val[:,-1]
    y_train = y_train[:,:-1]
    y_val= y_val[:,:-1]

    x_train_lbn = x_train[:,-len(parameters.LBN_inputs):].reshape(-1,4,len(parameters.LBN_inputs)//4)
    x_train = x_train[:,:-len(parameters.LBN_inputs)]

    x_val_lbn = x_val[:,-len(parameters.LBN_inputs):].reshape(-1,4,len(parameters.LBN_inputs)//4)
    x_val = x_val[:,:-len(parameters.LBN_inputs)]

    onehots = [getattr(OneHot,onehot)() for onehot in parameters.onehots]
    
    # Scaler #
    scaler_name = 'scaler_'+parameters.suffix+'_'.join(parameters.eras)+'.pkl' 
    with open(os.path.join(parameters.main_path,scaler_name), 'rb') as handle: # Import scaler that was created before
        scaler = pickle.load(handle)

    # Design network #

    # Left branch : classic inputs -> Preprocess -> onehot
    IN = Input(shape=x_train.shape[1:],name='IN')
    L0 = PreprocessLayer(batch_size=params['batch_size'],mean=scaler.mean_,std=scaler.scale_,name='Preprocess')(IN)
    OH = OneHot.OneHot(onehots=onehots,name="OneHot")(L0)

    # Right branch : LBN 
    INLBN = Input(shape=x_train_lbn.shape[1:],name='INLBN')
    FEATURES = LBNLayer(x_train_lbn.shape[1: ], 
                        n_particles = max(params['n_particles'],1), # Hack so that 0 does not trigger error
                        boost_mode  = LBN.PAIRS, 
                        features    = ["E", "px", "py", "pz", "pt", "p", "m", "pair_cos"],
                        name='LBN')(INLBN)
    BATCHNORM = tf.keras.layers.BatchNormalization(name='BATCHNORM')(FEATURES)
    # Concatenation of left and right #
    CONC = tf.keras.layers.Concatenate(axis=-1)([BATCHNORM, OH])
    L1 = Dense(params['first_neuron'],
               activation=params['activation'],
               kernel_regularizer=l2(params['l2']))(CONC if params['n_particles'] > 0 else OH)
    HIDDEN = hidden_layers(params,1,batch_normalization=True).API(L1)
    OUT = Dense(y_train.shape[1],activation=params['output_activation'],name='OUT')(HIDDEN)

    # Check preprocessing #
    preprocess = Model(inputs=[IN],outputs=[L0])
    out_preprocess = preprocess.predict(x_train,batch_size=params['batch_size'])
    mean_scale = np.mean(out_preprocess[:,[not m for m in parameters.mask_onehot]])
    std_scale = np.std(out_preprocess[:,[not m for m in parameters.mask_onehot]])
    if abs(mean_scale)>0.01 or abs((std_scale-1)/std_scale)>0.1: # Check that scaling is correct to 1%
        raise RuntimeError("Something is wrong with the preprocessing layer (mean = %0.6f, std = %0.6f), maybe you loaded an incorrect scaler"%(mean_scale,std_scale))

    # Tensorboard logs #
    #path_board = os.path.join(parameters.main_path,"TensorBoard")
    #suffix = 0
    #while(os.path.exists(os.path.join(path_board,"Run_"+str(suffix)))):
    #    suffix += 1
    #path_board = os.path.join(path_board,"Run_"+str(suffix))
    #os.makedirs(path_board)
    #logging.info("TensorBoard log dir is at %s"%path_board)

    # Callbacks #
    # Early stopping to stop learning if val_loss plateau for too long #
    early_stopping = EarlyStopping(**parameters.early_stopping_params)
    # Reduce learnign rate in case of plateau #
    reduceLR = ReduceLROnPlateau(**parameters.reduceLR_params)
    # Custom loss function plot for debugging #
    loss_history = LossHistory()
    # Tensorboard for checking live the loss curve #
    #board = TensorBoard(log_dir=path_board, 
    #                    histogram_freq=1, 
    #                    batch_size=params['batch_size'], 
    #                    write_graph=True, 
    #                    write_grads=True, 
    #                    write_images=True)
    Callback_list = [loss_history,early_stopping,reduceLR]

    # Compile #
    if 'resume' not in params:  # Normal learning 
        # Define model #
        model_inputs = [IN]
        if params['n_particles'] > 0:
            model_inputs.append(INLBN)
        model = Model(inputs=model_inputs, outputs=[OUT])
        initial_epoch = 0
    else: # a model has to be imported and resumes training
        custom_objects =  {'PreprocessLayer': PreprocessLayer,'OneHot': OneHot.OneHot}
        logging.info("Loaded model %s"%params['resume'])
        a = Restore(params['resume'],custom_objects=custom_objects,method='h5')
        model = a.model
        initial_epoch = params['initial_epoch']

        
    model.compile(optimizer=Adam(lr=params['lr']),
                  loss={'OUT':params['loss_function']},
                  metrics=['accuracy','AUC'])
    print (model.summary())
    fit_inputs = {'IN':x_train}
    fit_val = ({'IN':x_val},{'OUT':y_val},w_val)
    if params['n_particles'] > 0:
        fit_inputs['INLBN'] = x_train_lbn
        fit_val[0]['INLBN'] = x_val_lbn
    # Fit #
    history = model.fit(x               = fit_inputs,
                        y               = {'OUT':y_train},
                        sample_weight   = w_train,
                        epochs          = params['epochs'],
                        batch_size      = params['batch_size'],
                        verbose         = 1,
                        validation_data = fit_val,
                        callbacks       = Callback_list)

    # Plot history #
    PlotHistory(loss_history)

    return history,model



#################################################################################################
# NeuralNetGeneratorModel#
#################################################################################################
def NeuralNetGeneratorModel(x_train,y_train,x_val,y_val,params):
    """
    Keras model for the Neural Network, used to scan the hyperparameter space by Talos
    Uses the generator rather than the input data (which are dummies)
    """
    onehots = [getattr(OneHot,onehot)() for onehot in parameters.onehots]
    # Design network #
    scaler_name = 'scaler_'+parameters.suffix+'_'.join(parameters.eras)+'.pkl' 
    with open(os.path.join(parameters.main_path,scaler_name), 'rb') as handle: # Import scaler that was created before
        scaler = pickle.load(handle)
    IN = Input(shape=(x_train.shape[1],),name='IN')
    L0 = PreprocessLayer(batch_size=params['batch_size'],mean=scaler.mean_,std=scaler.scale_,name='Preprocess')(IN)
    OH = OneHot.OneHot(onehots=onehots)(L0)
    L1 = Dense(params['first_neuron'],
               activation=params['activation'],
               kernel_regularizer=l2(params['l2']))(OH)
    HIDDEN = hidden_layers(params,1,batch_normalization=True).API(L1)
    OUT = Dense(y_train.shape[1],activation=params['output_activation'],name='OUT')(HIDDEN)

    # Tensorboard logs #
#    path_board = os.path.join(parameters.main_path,"TensorBoard")
#    suffix = 0
#    while(os.path.exists(os.path.join(path_board,"Run_"+str(suffix)))):
#        suffix += 1
#    path_board = os.path.join(path_board,"Run_"+str(suffix))
#    os.makedirs(path_board)
#    logging.info("TensorBoard log dir is at %s"%path_board)

    # Callbacks #
    # Early stopping to stop learning if val_loss plateau for too long #
    early_stopping = EarlyStopping(**parameters.early_stopping_params)
    # Reduce learnign rate in case of plateau #
    reduceLR = ReduceLROnPlateau(**parameters.reduceLR_params)
    # Custom loss function plot for debugging #
    loss_history = LossHistory()
    # Tensorboard for checking live the loss curve #
#    board = TensorBoard(log_dir=path_board, 
#                        histogram_freq=1, 
#                        batch_size=params['batch_size'], 
#                        write_graph=True, 
#                        write_grads=True, 
#                        write_images=True)
#    Callback_list = [loss_history,early_stopping,reduceLR,board]
    Callback_list = [loss_history,early_stopping,reduceLR]

    # Compile #
    if 'resume' not in params:  # Normal learning 
        # Define model #
        model = Model(inputs=[IN], outputs=[OUT])
        initial_epoch = 0
    else: # a model has to be imported and resumes training
        custom_objects =  {'PreprocessLayer': PreprocessLayer}
        logging.info("Loaded model %s"%params['resume'])
        a = Restore(params['resume'],custom_objects=custom_objects,method='h5')
        model = a.model
        initial_epoch = params['initial_epoch']
        
    model.compile(optimizer=Adam(lr=params['lr']),
                  loss={'OUT':params['loss_function']},
                  metrics=['accuracy'])

    utils.print_summary(model=model) #used to print model
    # Generator #
    training_generator = DataGenerator(path = parameters.config,
                                       inputs = parameters.inputs,
                                       outputs = parameters.outputs,
                                       cut = parameters.cut,
                                       weight  = parameters.weight,
                                       batch_size = params['batch_size'],
                                       state_set = 'training',
                                       model_idx = params['model_idx'] if parameters.crossvalidation else None)
    validation_generator = DataGenerator(path = parameters.config,
                                         inputs = parameters.inputs,
                                         outputs = parameters.outputs,
                                         cut = parameters.cut,
                                         weight  = parameters.weight,
                                         batch_size = params['batch_size'],
                                         state_set = 'validation',
                                         model_idx = params['model_idx'] if parameters.crossvalidation else None)

    # Some verbose logging #
    logging.info("Will use %d workers"%parameters.workers)
    logging.warning("Tensorflow location "+ tf.__file__)
    logging.warning("GPU ")
    logging.warning(K.tensorflow_backend._get_available_gpus())
    # Fit #
    history = model.fit_generator(generator             = training_generator,   # Training data from generator instance
                                  validation_data       = validation_generator, # Validation data from generator instance
                                  epochs                = params['epochs'],     # Number of epochs
                                  verbose               = 1,
                                  max_queue_size        = parameters.workers*2, # Length of batch queue
                                  callbacks             = Callback_list,        # Callbacks
                                  initial_epoch         = initial_epoch,        # In case of resumed training will be different from 0
                                  workers               = parameters.workers,   # Number of threads for batch generation (0 : all in same)
                                  shuffle               = True,                 # Shuffle order at each epoch
                                  use_multiprocessing   = True)                 # Needs to be turned on for queuing batches
                                
    # Plot history #
    PlotHistory(loss_history)

    return history,model


