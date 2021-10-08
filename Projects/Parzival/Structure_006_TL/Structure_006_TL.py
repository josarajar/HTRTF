#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 11:43:59 2017

@author: aradillas
"""

import tensorflow as tf

import sys
sys.path.append('../../../Modules')
from utils import load_dataset, initialize_log, check_valid_and_test_sets, seconds_to_days_hours_min_sec
from tasks import train, validation, transfer
from math import ceil
import time

from layers import bidirectionalLSTM, max_pool, CNN, FNN

from shutil import copyfile       

from sys import float_info
eps=float_info.epsilon

    
class Network:
    
    def create(self, imageHeight,imageWidth, num_classes,evalFLAG):
        graph = tf.Graph()
        with graph.as_default():
            
            num_hidden = 256
            
            training = not evalFLAG

            with tf.name_scope('Inputs'):
                inputs = tf.placeholder(tf.float32, [None, imageHeight, imageWidth,1],name='inputs')
                if evalFLAG:
                    tf.summary.image('inputs',inputs,max_outputs=1)
    
            #seq_len should be feed with a list containing the real width of the images before padding to obtain imageWidth
            seq_len = tf.placeholder(tf.int32, [None],name='seq_len')
    
            targets = tf.sparse_placeholder(tf.int32, name='targets')
            
            targets_len = tf.placeholder(tf.int32, name='targets_len')    
      
            conv_keep_prob = 0.8
            lstm_keep_prob = 0.5


    
            # Layer 1
            with tf.name_scope('Layer_Conv_1'):
                h_conv1 = CNN(x=inputs, filters=16, kernel_size=[3,3], strides=[1,1], name='conv1', activation=tf.nn.leaky_relu, evalFLAG=evalFLAG, initializer=tf.contrib.layers.xavier_initializer(uniform=False))
                h_pool1, seq_len_1, imageHeight, imageWidth = max_pool(h_conv1,[2,2],seq_len, imageHeight, imageWidth, evalFLAG)
                h_pool1 = tf.layers.dropout(h_pool1,rate=0.0,training=training)
                
            
            # Layer 2
            with tf.name_scope('Layer_Conv_2'):
                h_conv2 = CNN(x=h_pool1, filters=32, kernel_size=[3,3], strides=[1,1], name='conv2', activation=tf.nn.leaky_relu, evalFLAG=evalFLAG, initializer=tf.contrib.layers.xavier_initializer(uniform=False))
                h_pool2, seq_len_2, imageHeight, imageWidth = max_pool(h_conv2,[2,2],seq_len_1, imageHeight, imageWidth, evalFLAG)
                h_pool2 = tf.layers.dropout(h_pool2,rate=(1-conv_keep_prob),training=training)
                
                             
            # Layer 3
            with tf.name_scope('Layer_Conv_3'):
                h_conv3 = CNN(x=h_pool2, filters=48, kernel_size=[3,3], strides=[1,1], name='conv3', activation=tf.nn.leaky_relu, evalFLAG=evalFLAG, initializer=tf.contrib.layers.xavier_initializer(uniform=False))
                h_pool3, seq_len_3, imageHeight, imageWidth = max_pool(h_conv3,[2,2],seq_len_2, imageHeight, imageWidth, evalFLAG)
                h_pool3 = tf.layers.dropout(h_pool3,rate=(1-conv_keep_prob),training=training)    
    
            # Layer 4
            with tf.name_scope('Layer_Conv_4'):
                h_conv4 = CNN(x=h_pool3, filters=64, kernel_size=[3,3], strides=[1,1], name='conv4', activation=tf.nn.leaky_relu, evalFLAG=evalFLAG, initializer=tf.contrib.layers.xavier_initializer(uniform=False))
                h_pool4, seq_len_4, imageHeight, imageWidth = max_pool(h_conv4,[1,1],seq_len_3, imageHeight, imageWidth, evalFLAG)
                h_pool4 = tf.layers.dropout(h_pool4,rate=(1-conv_keep_prob),training=training)     
                     
    
            # Layer 5
            with tf.name_scope('Layer_Conv_5'):
                h_conv5 = CNN(x=h_pool4, filters=80, kernel_size=[3,3], strides=[1,1], name='conv5', activation=tf.nn.leaky_relu, evalFLAG=evalFLAG, initializer=tf.contrib.layers.xavier_initializer(uniform=False))                
                h_pool5, seq_len_5, imageHeight, imageWidth = max_pool(h_conv5,[1,1],seq_len_4, imageHeight, imageWidth, evalFLAG)
                h_pool5 = tf.layers.dropout(h_pool5,rate=(1-lstm_keep_prob),training=training)
            
            with tf.name_scope('Reshaping_step'):
                h_cw_concat=tf.transpose(h_pool5, (2,0,1,3))
                h_cw_concat=tf.reshape(h_cw_concat, (int(imageWidth),-1,int(imageHeight*80)))
                h_cw_concat=tf.transpose(h_cw_concat, (1,0,2))
            
                
            with tf.name_scope('Layer_BLSTM_1'):
                
                h_bilstm1 = bidirectionalLSTM(h_cw_concat, num_hidden, seq_len_5, '1', evalFLAG)
                h_bilstm1 = tf.concat(h_bilstm1,2)
                h_bilstm1 = tf.layers.dropout(h_bilstm1,rate=(1-lstm_keep_prob),training=training)
                
            with tf.name_scope('Layer_BLSTM_2'):
                
                h_bilstm2 = bidirectionalLSTM(h_bilstm1, num_hidden, seq_len_5, '2', evalFLAG)
                h_bilstm2 = tf.concat(h_bilstm2,2)
                h_bilstm2 = tf.layers.dropout(h_bilstm2,rate=(1-lstm_keep_prob),training=training)

            with tf.name_scope('Layer_BLSTM_3'):
                
                h_bilstm3 = bidirectionalLSTM(h_bilstm2, num_hidden, seq_len_5, '3', evalFLAG)
                h_bilstm3 = tf.concat(h_bilstm3,2)
                h_bilstm3 = tf.layers.dropout(h_bilstm3,rate=(1-lstm_keep_prob),training=training)
                
            with tf.name_scope('Layer_BLSTM_4'):
                
                h_bilstm4 = bidirectionalLSTM(h_bilstm3, num_hidden, seq_len_5, '4', evalFLAG)
                h_bilstm4 = tf.concat(h_bilstm4,2)
                h_bilstm4 = tf.layers.dropout(h_bilstm4,rate=(1-lstm_keep_prob),training=training)

            with tf.name_scope('Layer_BLSTM_5'):
                
                h_bilstm5 = bidirectionalLSTM(h_bilstm4, num_hidden, seq_len_5, '5', evalFLAG)
                h_bilstm5 = tf.concat(h_bilstm5,2)
                h_bilstm5 = tf.layers.dropout(h_bilstm5,rate=(1-lstm_keep_prob),training=training)

            with tf.name_scope('Layer_Linear') as ns:
                outputs=tf.transpose(h_bilstm5, (1,0,2))        
                outputs=tf.reshape(outputs, (-1,2*num_hidden))   
                logits = FNN(outputs, num_classes, ns, None, evalFLAG)
                                
            with tf.name_scope('Logits'):
                logits = tf.reshape(logits, (int(imageWidth),-1,num_classes))
            
            seq_len_5=tf.maximum(seq_len_5,targets_len)
            
            n_batches=tf.placeholder(tf.float32,name='n_batches')
            previousCost=tf.placeholder(tf.float32, name='previous_cost')
            
            with tf.name_scope('CTC_Loss'):
                loss = tf.nn.ctc_loss(targets, logits, seq_len_5, preprocess_collapse_repeated=False, ctc_merge_repeated=True)
                with tf.name_scope('total'):
                    batch_cost = tf.reduce_mean(loss)
                    cost=batch_cost/n_batches+previousCost
                    
            tf.summary.scalar('CTC_loss', cost)
                
            with tf.name_scope('train'):
                train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Layer_Linear') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='BLSTM[12345]') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='conv[12345]')
                print(train_vars)
                learning_rate=tf.placeholder(tf.float32,name='learning_rate')
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(batch_cost)
     
            with tf.name_scope('predictions'):         
                predictions, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len_5, merge_repeated=False)
                
            with tf.name_scope('CER'):
                with tf.name_scope('Mean_CER_per_word'):
                    previousEDnorm = tf.placeholder(tf.float32,name='previousEDnorm')
                    EDnorm = tf.reduce_mean(tf.edit_distance(tf.cast(predictions[0], tf.int32), targets, normalize=True))/n_batches+previousEDnorm
                    
                    if evalFLAG:
                        tf.summary.scalar('EDnorm', EDnorm)
                        
                with tf.name_scope('Absolute_CER_total_set'):
                    setTotalChars=tf.placeholder(tf.float32, name='setTotalChars')
                    previousEDabs=tf.placeholder(tf.float32, name='previousEDabs')
                    errors=tf.edit_distance(tf.cast(predictions[0], tf.int32), targets, normalize=False)
                    EDabs = tf.reduce_sum(errors)/setTotalChars+previousEDabs
                    if evalFLAG:
                        tf.summary.scalar('EDabs', EDabs)  
                    
            
            ED=[EDnorm, EDabs]
            
            saver=tf.train.Saver(tf.global_variables(),max_to_keep=5,keep_checkpoint_every_n_hours=24)
            
            transferred_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="BLSTM[12345]") + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="conv")

            transferred_vars_dict = dict([(var.op.name, var) for var in transferred_vars])

            transfer_saver = tf.train.Saver(transferred_vars_dict)

            merged = tf.summary.merge_all()
            
            return graph, [saver, transfer_saver], inputs, seq_len, targets, targets_len, learning_rate, n_batches, setTotalChars, previousEDabs, previousEDnorm, previousCost, optimizer, batch_cost,  cost, errors, ED,  predictions, merged


def main():
    ## Sets
    train_set_path='../Sets/reduced350/demo.h5'
    validation_set_path='../Sets/reduced350/demo.h5'
    test_set_path='../Sets/reduced350/demoo.h5'
    
#    train_set_path='../Sets/reduced350/test.h5'
#    validation_set_path='../Sets/reduced350/valid.h5'
#    test_set_path='../Sets/reduced350/train.h5'
                
    ## Database location
    DataBasePath='../Database/parzivaldb-v1.0/data/line_images_normalized/'
    
    
    import datetime
    now=now = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    
    transferFLAG = True
    model_for_transfer_path='./model_for_transfer'
           
    testFLAG = False
    
    batch_size = 20 
    num_epochs  = 250
    learning_rate = 0.0003
    num_epochs_before_validation = 10
    
    restore_checkpoint_at_epoch = 0

    ## Log file
    if testFLAG:
       indicator = sys.argv[0].split('/')[-1].split('-')[0]
       files_path='./train-{}/eval-{}/'.format(indicator, now)
       log_path='./train-{}/eval-{}/log/'.format(indicator, now)
       log_file_path='./train-{}/eval-{}/log/log.txt'.format(indicator, now)
       models_path='./train-{}/models/'.format(indicator)
       TensorBoard_dir='./train-{}/eval-{}/TensorBoard_files/'.format(indicator, now)
       tf.gfile.MakeDirs(files_path)  
       tf.gfile.MakeDirs(log_path)
       tf.gfile.MakeDirs(TensorBoard_dir)
       copyfile(sys.argv[0],files_path+'{}-'.format(now)+sys.argv[0])
       log_file_indicator=initialize_log(log_file_path, mode='w')
    elif restore_checkpoint_at_epoch==0 or transferFLAG: 
       files_path='./train-{}/'.format(now)
       log_path='./train-{}/log'.format(now)
       log_file_path='./train-{}/log/log.txt'.format(now)
       models_path='./train-{}/models/'.format(now)
       TensorBoard_dir='./train-{}/TensorBoard_files/'.format(now)
       tf.gfile.MakeDirs(files_path)  
       tf.gfile.MakeDirs(log_path)
       tf.gfile.MakeDirs(models_path)
       tf.gfile.MakeDirs(TensorBoard_dir)
       copyfile(sys.argv[0],files_path+'{}-'.format(now)+sys.argv[0])
       copyfile(sys.argv[0],'./{}-'.format(now)+sys.argv[0])
       log_file_indicator=initialize_log(log_file_path, mode='w')
    else:
       indicator = sys.argv[0].split('/')[-1].split('-')[0]
       log_path='./train-{}/log'.format(indicator)
       log_file_path='./train-{}/log/log.txt'.format(indicator)
       models_path='./train-{}/models/'.format(indicator)
       TensorBoard_dir='./train-{}/TensorBoard_files/'.format(indicator)
       log_file_indicator=initialize_log(log_file_path,mode='a')
       log_file_indicator.write(('#'*100+'\n')*5+'\n\nRecovering after break or pause in epoch '+str(restore_checkpoint_at_epoch)+'\n\n'+('#'*100+'\n')*5)

    
    num_steps=ceil(num_epochs/num_epochs_before_validation)          
    validSet, valid_imageHeight, valid_imageWidth, valid_labels = load_dataset(validation_set_path,DataBasePath, log_file_indicator, database = 'Parzival')
    
    if not testFLAG:
       trainSet, train_imageHeight, train_imageWidth, train_labels = load_dataset(train_set_path,DataBasePath, log_file_indicator, database='Parzival')
       imageHeight, labels = check_valid_and_test_sets(train_imageHeight, valid_imageHeight, train_imageHeight, train_labels, valid_labels, train_labels, log_file_indicator)
       train_writer = tf.summary.FileWriter(TensorBoard_dir+'train_task')
       valid_vs_writer = tf.summary.FileWriter(TensorBoard_dir+'valid_task_validset')
       valid_ts_writer = tf.summary.FileWriter(TensorBoard_dir+'valid_task_trainset')   
    else:
       testSet,  test_imageHeight, test_imageWidth, test_labels = load_dataset(test_set_path,DataBasePath, log_file_indicator, database = 'Parzival')
       imageHeight, labels = check_valid_and_test_sets(test_imageHeight, valid_imageHeight, test_imageHeight, test_labels, valid_labels, test_labels, log_file_indicator)
       test_writer = tf.summary.FileWriter(TensorBoard_dir+'test_validset')
       valid_writer = tf.summary.FileWriter(TensorBoard_dir+'valid_testset')
    log_file_indicator.flush()
    
    
    
    # The number of classes is the amount of labels plus 1 for blanck
    num_classes=len(labels)+1
    
    train_start=time.time()
    network_train=Network()
    if transferFLAG:
        epoch = restore_checkpoint_at_epoch
        transfer(epoch, network_train,  imageHeight, train_imageWidth, num_classes, log_file_indicator, model_for_transfer_path, models_path, train_writer)
    
    if not testFLAG:
        for step in range(ceil(restore_checkpoint_at_epoch/num_epochs_before_validation), num_steps):
                
           train(step, network_train, num_epochs_before_validation, batch_size, learning_rate, trainSet, imageHeight, train_imageWidth, num_classes, log_file_indicator, models_path, train_writer, transferFLAG)
        
           epoch=(step+1)*num_epochs_before_validation-1
        
           validation(epoch, network_train, batch_size,'validation', validSet,imageHeight,valid_imageWidth,labels, num_classes,log_file_indicator, models_path, valid_vs_writer)
           validation(epoch, network_train, batch_size,'train', trainSet,imageHeight,train_imageWidth,labels, num_classes,log_file_indicator, models_path, valid_ts_writer)

        train_end=time.time()
        train_duration=train_end-train_start
        print('Training completed in: '+seconds_to_days_hours_min_sec(train_duration))
        log_file_indicator.write('\nTraining completed in: '+seconds_to_days_hours_min_sec(train_duration, day_flag=True)+'\n')

    else:
        
        epoch = restore_checkpoint_at_epoch 
        text='\nEvaluating model at epoch {}...\n'.format(epoch)
        print(text)
        log_file_indicator.write(text)        
        validation(epoch, network_train, batch_size,'validation', validSet,imageHeight,valid_imageWidth,labels, num_classes,log_file_indicator, models_path, valid_writer)
        validation(epoch, network_train, batch_size,'test', testSet,imageHeight,test_imageWidth,labels, num_classes,log_file_indicator, models_path, test_writer)
        
        test_end=time.time()
        test_duration=test_end-train_start
        print('Evaluation completed in: '+seconds_to_days_hours_min_sec(test_duration))
        log_file_indicator.write('\nEvaluation completed in: '+seconds_to_days_hours_min_sec(test_duration, day_flag=True)+'\n')
    
    if not testFLAG:
      train_writer.close()
      valid_ts_writer.close()
      valid_vs_writer.close()
    else:
      test_writer.close()
      valid_writer.close()
    
    
    log_file_indicator.flush()
    log_file_indicator.close()
    
if __name__ == '__main__':
    main()

