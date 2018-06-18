import sys

def initialize_log(file_path, mode='a'):
    """
    Initialize a file in which logs will be written during the execution of 
    training, validation...
    
    Args:
        file_path: string with the path to the file
        mode: default 'a', (if the file exists, it will continue writing logs
            after the last line.). 'w' is also possible if you want to create 
            blank file.
        
        
    Returns:
        _io.TextIOWrapper of the file where logs will be written
        
        
    Raises:
        IOError if the path doesn't exists. 
    """
    try:
        log_file=open(file_path, mode)
        return log_file
    except IOError:
        print('It doesn\'t exit path: ',file_path)
        raise

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]     
    
def target_list_to_sparse_tensor(targetList):

    """
    Make tensorflow SparseTensor from list of targets, with each element in 
    the list being a list or array with the values of the target sequence
    
    Args:
        targetList: list of target sequences      
        
    Returns:
        tuple containing three elements: array of indices, array of vals and 
            array containing the shape 
        
    """
    import numpy as np
    indices = []
    vals = []
    for tI, target in enumerate(targetList):
        for seqI, val in enumerate(target):
            indices.append([tI, seqI])
            vals.append(val)
    shape = [len(targetList), np.asarray(indices).max(0)[1]+1]
    return (np.array(indices), np.array(vals), np.array(shape))



def load_dataset(h5_file, DataBasePath, log_file_indicator, x_pad=0, y_pad=0, normalization=True, resize = True, padding = True, database='IAM'):
    """
    Make dataset by loading input images and creating some lists from the csv
    file which contains information about dataset, such as image names, target
    sequences, target transcriptions and image sizes.
    
    Args:
        h5_file: path to the file which contains the dataset information.
        DataBasePath: default path to the folder which contains the images.
        log_file_indicator: _io.TextIOWrapper to write the logs.
        x_pad: padding added to the images in the horizontal axis if neccesary.
        y_pad: padding added to the images in the vertical axis if neccesary.
        normalization: boolean indicating if the image pixels values are rescaled
            between (0-1).
        resize: bool indicating if the images are resized so that every image
            has the same height to feed the network.
        padding: boolean indicating if the images are padded in order to have the
            same width of the larger one.
        database: string showing the database which is to be load, the IAM
            databasehas a specific directory structure. If another
            database, image files will be read from DataBasePath directly.
                   
        
    Returns:
        nameList: list of image names
        inputs: list of images in format numpy.ndarray
        targetList: list of secuences (numerical label)
        seqLengths: list of width of text images (before padding)
        seqHeights: list of heigths of text images
        transcriptionList: list of transcriptions (ascii)
        transcriptionLenList: list of transcriptions lengths
        imageHeight: height of images (unique value)
        imageWidth: width of images (unique value)
        labels: list of labels in the dataset
                       
    Raises:
        OSError if the csv file doesn't exist
        Error if some file Image doesn't exist
    """
    from PIL import Image
    import numpy as np
    import h5py
    from time import time
    from math import floor
    from sys import float_info
    
    eps=float_info.epsilon
    
    try:
        print('Loading dataset from h5 file: '+str(h5_file)+'. Info in: '+h5_file[:-3]+'.csv')
        log_file_indicator.write('\nLoading dataset from h5 file: '+str(h5_file)+'. Info in: '+str(h5_file.split('.')[-2])+'.csv\n')
        fh5=h5py.File(h5_file,'r')
    except OSError:
        print('file not found: ' + str(h5_file))
        log_file_indicator.write('\n Exception, file not found: ' + str(h5_file)+'\n')
        print('Closing')
        log_file_indicator.close()
        sys.exit(1)
    else:       
        nameList=[]
        inputs=[]
        targetList_aux=list(fh5['data/targets'])
        targetList=[]
        transcriptionList=[]
        seqLengths=[]
        seqHeights=[]
        transcriptionLenList=[]
        labels=dict()
        for ind, label in enumerate(fh5['labels']):
            labels[ind]=label
        
        prev_percent=-1
        set_size=len(fh5['data/imgNames'])
        init_time = time()
        imageHeight=max(fh5['data/sizes'][:,1])+y_pad*2
        imageWidth=max(fh5['data/sizes'][:,0])+x_pad*2
        for i in range(set_size):
            img_name=fh5['data/imgNames'][i]
            nameList.append(img_name)
            img_name_split=img_name.split('-')
            if database == 'IAM':
                img_path=str.rstrip(DataBasePath,'/')+'/'+img_name_split[0]+'/'+img_name_split[0]+'-'+img_name_split[1]+'/'+img_name+'.png'
            else:
                img_path=str.rstrip(DataBasePath,'/')+'/'+img_name+'.png'
            try:
                img = Image.open(img_path)
            except:
                print('File not found: '+str(img_path))
                log_file_indicator.write('\n Exception, file not found: ' + str(img_path)+'\n')
                log_file_indicator.close()
                sys.exit(1)
            else:
                ancho_init, alto_init = img.size
                size=fh5['data/sizes'][i]
                if resize:
                    img = img.resize(size, Image.LANCZOS)
                    img_numpy=np.array(img.getdata()).reshape(size[1],size[0])                 
                    if padding:
                        img_numpy=np.pad(img_numpy,((y_pad,y_pad+max(fh5['data/sizes'][:,1])-size[1]),(x_pad,x_pad+max(fh5['data/sizes'][:,0])-size[0])),mode='constant', constant_values=255)
                        img_numpy=img_numpy.reshape(imageHeight,imageWidth,1)
                        seqLengths.append(size[0]+x_pad)
                        seqHeights.append(size[1]+y_pad)
                    else:
                       seqLengths.append(size[0])
                       seqHeights.append(size[1]) 
                else:
                    img_numpy=np.array(img.getdata()).reshape(alto_init, ancho_init)
                    seqLengths.append(ancho_init)
                    seqHeights.append(alto_init)
                img_numpy=(255-img_numpy)
                if normalization:
                    img_numpy=img_numpy/255
                time_elapsed=floor(1000*(time()-init_time))/1000
                
                if floor(100*(i+1)/set_size) > prev_percent:
                    prev_percent=floor(10000*(i+1)/set_size)/100
                    remaining_time=floor(1000*(100*(time_elapsed+eps)/(prev_percent+eps)-(time_elapsed+eps)))/1000
                    print('Loading dataset from h5 file: '+str(h5_file)+'.\nTime elapsed: '+seconds_to_days_hours_min_sec(time_elapsed) +'. Remaining time: ' +seconds_to_days_hours_min_sec(remaining_time) + '.\n')
                    print('['+int(prev_percent)*'|'+(100-int(prev_percent))*' '+'] '+str(prev_percent)+'%\n')
                
                inputs.append(img_numpy)
                
                
                transcriptionLenList.append(fh5['data/trans_len'][i])
                transcriptionList.append(fh5['data/transcriptions'][i])
                targetList.append(targetList_aux[0:transcriptionLenList[-1]])
                del(targetList_aux[0:transcriptionLenList[-1]])
        
        log_file_indicator.write('\n'+str(set_size)+' sequences loaded.\n')
        print(str(len(nameList))+' sequences loaded.')
        return [nameList, inputs, targetList, seqLengths, seqHeights, transcriptionList, transcriptionLenList], imageHeight, imageWidth, labels

    
def check_valid_and_test_sets(train_imageHeight, valid_imageHeight, test_imageHeight, train_labels, valid_labels, test_labels, log_file_indicator):
    """
    It checks if the height of the images in every set (train, validation and test) has the same height, in case the structure requires it. It
    also checks if the labels are the same in order.
    
    Args:
        train_imageHeight: height of images in the train set
        valid_imageHeight: height of images in the validation set
        test_imageHeight: height of images in the test set
        train_labels: list of labels in the train set
        valid_labels: list of labels in the validation set
        test_labels: list of labels in the test set
        log_file_indicator: _io.TextIOWrapper of the file where logs will be written
        
    Returns:
        train_imageHeight: the height of the images when all of them are the same.
        train_labels: the list of labels if they are the same in all datasets.
    
    Exits:
        when the height and labels are not the same in all datasets.
        
    """
    if valid_imageHeight==train_imageHeight:
        log_file_indicator.write('\n The height of images in valid set are the same of the ones in train set: '+str(train_imageHeight) + 'px. \n')
    else:
        log_file_indicator.write('\n The height of images in valid set ('+str(valid_imageHeight)+'px.) are different of the ones in train set: ('+str(train_imageHeight)+'px.) \n' )
        print('\n The height of images in valid set ('+str(valid_imageHeight)+'px.) are different of the ones in train set: ('+str(train_imageHeight)+'px.) \n')
        log_file_indicator.close()
        sys.exit(1)
        
    if test_imageHeight==train_imageHeight:
        log_file_indicator.write('\n The height of images in test set are the same of the ones in train set: '+str(train_imageHeight) + 'px. \n')
    else:
        log_file_indicator.write('\n The height of images in test set ('+str(test_imageHeight)+'px.) are different of the ones in train set: ('+str(train_imageHeight)+'px.) \n' )
        print('\n The height of images in test set ('+str(test_imageHeight)+'px.) are different of the ones in train set: ('+str(train_imageHeight)+'px.) \n')
        log_file_indicator.close()
        sys.exit(1)
        
    if valid_labels==train_labels:
        log_file_indicator.write('\n The labels in valid set are the same of the ones in train set: '+str(train_labels) + '. \n')
    else:
        log_file_indicator.write('\n The labels in valid set ('+str(valid_labels)+'px.) are different of the ones in train set: ('+str(train_labels)+'px.) \n' )
        print('\n The labels in valid set ('+str(valid_labels)+'px.) are different of the ones in train set: ('+str(train_labels)+'px.) \n')
        log_file_indicator.close()
        sys.exit(1)
        
    if test_labels==train_labels:
        log_file_indicator.write('\n The labels in test set are the same of the ones in train set: '+str(train_labels) + '. \n')
    else:
        log_file_indicator.write('\n The labels in test set ('+str(test_labels)+'px.) are different of the ones in train set: ('+str(train_labels)+'px.) \n' )
        print('\n The labels in test set ('+str(test_labels)+'px.) are different of the ones in train set: ('+str(train_labels)+'px.) \n')
        log_file_indicator.close()
        sys.exit(1)
    
    return train_imageHeight, train_labels

def shuffle_set(nameList, inputs, targetList, seqLengths, heights, transcriptionList, transLenList):
    """
    It receives lists containing different features over the dataset and shuffles them which the aim
    of training minibatches not having the same elements.
    
    Args:
        nameList: list of image names
        inputs: list of images in format numpy.ndarray
        targetList: list of secuences (numerical label)
        seqLengths: list of width of text images (before padding)
        heights: list of heigths of text images
        transcriptionList: list of transcriptions (ascii)
        transLenList: list of transcriptions lengths
        
    Returns:
        nameList_shuffled: list of image names
        inputs_shuffled: list of images in format numpy.ndarray
        targetList_shuffled: list of secuences (numerical label)
        seqLengths_shuffled: list of width of text images (before padding)
        heights_shuffled: list of heigths of text images
        transcriptionList_shuffled: list of transcriptions (ascii)
        transLenList_shuffled: list of transcriptions lengths

    """
    from random import shuffle, random
    
    nameList_shuffled=list(nameList)
    inputs_shuffled=list(inputs)
    targetList_shuffled=list(targetList)
    seqLengths_shuffled=list(seqLengths)
    heights_shuffled=list(heights)
    transcriptionList_shuffled=list(transcriptionList)
    transLenList_shuffled=list(transLenList)
    
    r = random()            # randomly generating a real in [0,1)
    shuffle(nameList_shuffled, lambda : r)  # lambda : r is an unary function which returns r
    shuffle(inputs_shuffled, lambda : r)  # using the same function as used in prev line so that shuffling order is same 
    shuffle(targetList_shuffled, lambda : r)
    shuffle(seqLengths_shuffled, lambda : r)
    shuffle(heights_shuffled, lambda : r)
    shuffle(transcriptionList_shuffled, lambda : r)
    shuffle(transLenList_shuffled, lambda : r)
    
    return nameList_shuffled, inputs_shuffled, targetList_shuffled, seqLengths_shuffled, heights_shuffled, transcriptionList_shuffled, transLenList_shuffled
    
def get_batch(batchSize, nameList, inputs, targetList, seqLengths, heights, transcriptionList, transLenList):
    """
    It receives lists containing different features over the dataset get a batch of
    'batchSize' elements.
    
    Args:
        batchSize: size of the batch which will be extracted.
        nameList: list of image names
        inputs: list of images in format numpy.ndarray
        targetList: list of secuences (numerical label)
        seqLengths: list of width of text images (before padding)
        heights: list of heigths of text images
        transcriptionList: list of transcriptions (ascii)
        transLenList: list of transcriptions lengths
        
    Returns:
        batchNameList: list of image names
        batchInputs: list of images in format numpy.ndarray
        batchTargetSparse: list of secuences (numerical label)
        batchSeqLengths: list of width of text images (before padding)
        batchHeights: list of heigths of text images
        batchTranscriptions: list of transcriptions (ascii)
        batchTransLen: list of transcriptions lengths

    """
    import numpy as np
    batchNameList=nameList[0:batchSize]
    del(nameList[0:batchSize])
    batchInputs=np.stack(inputs[0:batchSize], axis=0)
    del(inputs[0:batchSize])
    batchTargetSparse=target_list_to_sparse_tensor(targetList[0:batchSize])
    del(targetList[0:batchSize])
    batchSeqLengths=np.array([int(x) for x in seqLengths[0:batchSize]])
    del(seqLengths[0:batchSize])
    batchHeights=np.array([int(x) for x in heights[0:batchSize]])
    del(heights[0:batchSize])
    batchTranscriptions=transcriptionList[0:batchSize]
    del(transcriptionList[0:batchSize])
    batchTransLen=np.array([int(x) for x in transLenList[0:batchSize]])
    del(transLenList[0:batchSize])
    
    return batchNameList, batchInputs, batchTargetSparse, batchSeqLengths, batchHeights, batchTranscriptions, batchTransLen

def seconds_to_days_hours_min_sec(seconds, day_flag=False):
    """
    It recieves a number of seconds and returns a string showing the hours, 
    minutes and seconds. It also returns the number of days if day_flag is
    set to True
    
    Args:
       seconds: number of seconds
       day_flag: flag indicating if the returned string will contain the number of days.
           
    Returns:
        a string showing days, hours, minutes and seconds.
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if day_flag==True:
        d, h = divmod(h, 24)
        return "%d d. %d h. %d min. %0.2f s. " % (d,h,m,s)
    else:
        return "%d h. %d min. %0.2f s. " % (h,m,s)

def print_valid_results(epoch, set_name, SetSize, cost, ler, wer, log_indicator):
    """
    It receives an epoch, the name and size of the set, the cost and ler evaluated and show results in addition
    to being registered in log file.
    
    Args:
        epoch: the epoch number
        set_name: the name of the set
        SetSize: the size of the set
        cost: the cost function evaluated over the set
        ler: the Label Error Rate evaluated over the set
        log_indicator: _io.TextIOWrapper of the file where logs will be written
            
    """
    from math import ceil, floor
    results_text='\nEpoch '+str(epoch)+ ' results: \n' +'\n' +' '*floor((100-len(set_name+':'))/2)+set_name+':'+' '*ceil((100-len(set_name+':'))/2) + '\n' + 100*'-' + '\n' +  str(SetSize) + ' sequences evaluated.\nCost function: '+str(cost)+ ' \nMedium CER per sequence: ' +str(ler[0]) + '. \nTotal CER : ' +str(ler[1])+'.\n' +   '\n' + '. \nTotal WER : ' +str(wer)+'.\n' +   '\n'  
    print(results_text)             
    log_indicator.write(results_text)
    
def load_AACHEN_PARAMS(AHCHEN_h5_file, log_file_indicator):
    """
    This module extract parameters trainded with the framework https://github.com/rwth-i6/returnn
    and the neural network proposed in the mdlsmt demo structure.
    
    Args:
        AHCHEN_h5_file: file in format hdf5 generated with https://github.com/rwth-i6/returnn framework.
        log_file_indicator: _io.TextIOWrapper of the file where logs will be written
        
    Returns:
        The parameters of each layer of the network.
            
    """
    import h5py
    import numpy as np
    try:
        print('Loading AACHEN params from h5 file: '+str(AHCHEN_h5_file))
        log_file_indicator.write('\nLoading AACHEN params from h5 file: '+str(AHCHEN_h5_file)+'.\n')
        fh5=h5py.File(AHCHEN_h5_file,'r')
    except OSError:
        print('File not found: ' + str(AHCHEN_h5_file))
        log_file_indicator.write('\n Exception, file not found: ' + str(AHCHEN_h5_file)+'\n')
        print('Closing')
        log_file_indicator.close()
        sys.exit(1)
    else:
        
        w_conv0=fh5['conv0']['W_conv0'][:]
        w_conv0=w_conv0.transpose([2,3,1,0]) #AACHEN shape (n_out, n_in, filter_size), tensorflow shape (filter_size, n_in, n_out)
        b_conv0=fh5['conv0']['b_conv0'][:]
        
        w_conv1=fh5['conv1']['W_conv1'][:]
        w_conv1=w_conv1.transpose([2,3,1,0]) #AACHEN shape (n_out, n_in, filter_size), tensorflow shape (filter_size, n_in, n_out)
        b_conv1=fh5['conv1']['b_conv1'][:]
        
        w_conv2=fh5['conv2']['W_conv2'][:]
        w_conv2=w_conv2.transpose([2,3,1,0]) #AACHEN shape (n_out, n_in, filter_size), tensorflow shape (filter_size, n_in, n_out)
        b_conv2=fh5['conv2']['b_conv2'][:]
        
        w_conv3=fh5['conv3']['W_conv3'][:]
        w_conv3=w_conv3.transpose([2,3,1,0]) #AACHEN shape (n_out, n_in, filter_size), tensorflow shape (filter_size, n_in, n_out)
        b_conv3=fh5['conv3']['b_conv3'][:]
        
        w_conv4=fh5['conv4']['W_conv4'][:]
        w_conv4=w_conv4.transpose([2,3,1,0]) #AACHEN shape (n_out, n_in, filter_size), tensorflow shape (filter_size, n_in, n_out)
        b_conv4=fh5['conv4']['b_conv4'][:]
        
        
        # 1 AACHEN --> dwd-fwd
        V_x1_mdlstm0, V_h1_mdlstm0, V_v1_mdlstm0=fh5['mdlstm0']['W1_mdlstm0'][:], fh5['mdlstm0']['U1_mdlstm0'][:], fh5['mdlstm0']['V1_mdlstm0'][:]
        W_df_mdlstm0=np.concatenate((V_x1_mdlstm0,V_h1_mdlstm0, V_v1_mdlstm0), axis=0)       
        b_df_mdlstm0=fh5['mdlstm0']['b1_mdlstm0']
        
        # 2 AACHEN --> uwd-fwd
        V_x2_mdlstm0, V_h2_mdlstm0, V_v2_mdlstm0=fh5['mdlstm0']['W2_mdlstm0'][:], fh5['mdlstm0']['U2_mdlstm0'][:], fh5['mdlstm0']['V2_mdlstm0'][:]
        W_uf_mdlstm0=np.concatenate((V_x2_mdlstm0,V_h2_mdlstm0, V_v2_mdlstm0), axis=0)       
        b_uf_mdlstm0=fh5['mdlstm0']['b2_mdlstm0']
        
        # 2 AACHEN --> dwd-bwd
        V_x3_mdlstm0, V_h3_mdlstm0, V_v3_mdlstm0=fh5['mdlstm0']['W3_mdlstm0'][:], fh5['mdlstm0']['U3_mdlstm0'][:], fh5['mdlstm0']['V3_mdlstm0'][:]
        W_db_mdlstm0=np.concatenate((V_x3_mdlstm0,V_h3_mdlstm0, V_v3_mdlstm0), axis=0)       
        b_db_mdlstm0=fh5['mdlstm0']['b3_mdlstm0']
        
        # 2 AACHEN --> uwd-bwd
        V_x4_mdlstm0, V_h4_mdlstm0, V_v4_mdlstm0=fh5['mdlstm0']['W4_mdlstm0'][:], fh5['mdlstm0']['U4_mdlstm0'][:], fh5['mdlstm0']['V4_mdlstm0'][:]
        W_ub_mdlstm0=np.concatenate((V_x4_mdlstm0,V_h4_mdlstm0, V_v4_mdlstm0), axis=0)       
        b_ub_mdlstm0=fh5['mdlstm0']['b4_mdlstm0']
        
        
        # 1 AACHEN --> dwd-fwd
        V_x1_mdlstm1, V_h1_mdlstm1, V_v1_mdlstm1=fh5['mdlstm1']['W1_mdlstm1'][:], fh5['mdlstm1']['U1_mdlstm1'][:], fh5['mdlstm1']['V1_mdlstm1'][:]
        W_df_mdlstm1=np.concatenate((V_x1_mdlstm1,V_h1_mdlstm1, V_v1_mdlstm1), axis=0)       
        b_df_mdlstm1=fh5['mdlstm1']['b1_mdlstm1']
        
        # 2 AACHEN --> uwd-fwd
        V_x2_mdlstm1, V_h2_mdlstm1, V_v2_mdlstm1=fh5['mdlstm1']['W2_mdlstm1'][:], fh5['mdlstm1']['U2_mdlstm1'][:], fh5['mdlstm1']['V2_mdlstm1'][:]
        W_uf_mdlstm1=np.concatenate((V_x2_mdlstm1,V_h2_mdlstm1, V_v2_mdlstm1), axis=0)       
        b_uf_mdlstm1=fh5['mdlstm1']['b2_mdlstm1']
        
        # 2 AACHEN --> dwd-bwd
        V_x3_mdlstm1, V_h3_mdlstm1, V_v3_mdlstm1=fh5['mdlstm1']['W3_mdlstm1'][:], fh5['mdlstm1']['U3_mdlstm1'][:], fh5['mdlstm1']['V3_mdlstm1'][:]
        W_db_mdlstm1=np.concatenate((V_x3_mdlstm1,V_h3_mdlstm1, V_v3_mdlstm1), axis=0)       
        b_db_mdlstm1=fh5['mdlstm1']['b3_mdlstm1']
        
        # 2 AACHEN --> uwd-bwd
        V_x4_mdlstm1, V_h4_mdlstm1, V_v4_mdlstm1=fh5['mdlstm1']['W4_mdlstm1'][:], fh5['mdlstm1']['U4_mdlstm1'][:], fh5['mdlstm1']['V4_mdlstm1'][:]
        W_ub_mdlstm1=np.concatenate((V_x4_mdlstm1,V_h4_mdlstm1, V_v4_mdlstm1), axis=0)       
        b_ub_mdlstm1=fh5['mdlstm1']['b4_mdlstm1']
        
        
        
        # 1 AACHEN --> dwd-fwd
        V_x1_mdlstm2, V_h1_mdlstm2, V_v1_mdlstm2=fh5['mdlstm2']['W1_mdlstm2'][:], fh5['mdlstm2']['U1_mdlstm2'][:], fh5['mdlstm2']['V1_mdlstm2'][:]
        W_df_mdlstm2=np.concatenate((V_x1_mdlstm2,V_h1_mdlstm2, V_v1_mdlstm2), axis=0)       
        b_df_mdlstm2=fh5['mdlstm2']['b1_mdlstm2']
        
        # 2 AACHEN --> uwd-fwd
        V_x2_mdlstm2, V_h2_mdlstm2, V_v2_mdlstm2=fh5['mdlstm2']['W2_mdlstm2'][:], fh5['mdlstm2']['U2_mdlstm2'][:], fh5['mdlstm2']['V2_mdlstm2'][:]
        W_uf_mdlstm2=np.concatenate((V_x2_mdlstm2,V_h2_mdlstm2, V_v2_mdlstm2), axis=0)       
        b_uf_mdlstm2=fh5['mdlstm2']['b2_mdlstm2']
        
        # 2 AACHEN --> dwd-bwd
        V_x3_mdlstm2, V_h3_mdlstm2, V_v3_mdlstm2=fh5['mdlstm2']['W3_mdlstm2'][:], fh5['mdlstm2']['U3_mdlstm2'][:], fh5['mdlstm2']['V3_mdlstm2'][:]
        W_db_mdlstm2=np.concatenate((V_x3_mdlstm2,V_h3_mdlstm2, V_v3_mdlstm2), axis=0)       
        b_db_mdlstm2=fh5['mdlstm2']['b3_mdlstm2']
        
        # 2 AACHEN --> uwd-bwd
        V_x4_mdlstm2, V_h4_mdlstm2, V_v4_mdlstm2=fh5['mdlstm2']['W4_mdlstm2'][:], fh5['mdlstm2']['U4_mdlstm2'][:], fh5['mdlstm2']['V4_mdlstm2'][:]
        W_ub_mdlstm2=np.concatenate((V_x4_mdlstm2,V_h4_mdlstm2, V_v4_mdlstm2), axis=0)       
        b_ub_mdlstm2=fh5['mdlstm2']['b4_mdlstm2']
        
        
        
        # 1 AACHEN --> dwd-fwd
        V_x1_mdlstm3, V_h1_mdlstm3, V_v1_mdlstm3=fh5['mdlstm3']['W1_mdlstm3'][:], fh5['mdlstm3']['U1_mdlstm3'][:], fh5['mdlstm3']['V1_mdlstm3'][:]
        W_df_mdlstm3=np.concatenate((V_x1_mdlstm3,V_h1_mdlstm3, V_v1_mdlstm3), axis=0)       
        b_df_mdlstm3=fh5['mdlstm3']['b1_mdlstm3']
        
        # 2 AACHEN --> uwd-fwd
        V_x2_mdlstm3, V_h2_mdlstm3, V_v2_mdlstm3=fh5['mdlstm3']['W2_mdlstm3'][:], fh5['mdlstm3']['U2_mdlstm3'][:], fh5['mdlstm3']['V2_mdlstm3'][:]
        W_uf_mdlstm3=np.concatenate((V_x2_mdlstm3,V_h2_mdlstm3, V_v2_mdlstm3), axis=0)       
        b_uf_mdlstm3=fh5['mdlstm3']['b2_mdlstm3']
        
        # 2 AACHEN --> dwd-bwd
        V_x3_mdlstm3, V_h3_mdlstm3, V_v3_mdlstm3=fh5['mdlstm3']['W3_mdlstm3'][:], fh5['mdlstm3']['U3_mdlstm3'][:], fh5['mdlstm3']['V3_mdlstm3'][:]
        W_db_mdlstm3=np.concatenate((V_x3_mdlstm3,V_h3_mdlstm3, V_v3_mdlstm3), axis=0)       
        b_db_mdlstm3=fh5['mdlstm3']['b3_mdlstm3']
        
        # 2 AACHEN --> uwd-bwd
        V_x4_mdlstm3, V_h4_mdlstm3, V_v4_mdlstm3=fh5['mdlstm3']['W4_mdlstm3'][:], fh5['mdlstm3']['U4_mdlstm3'][:], fh5['mdlstm3']['V4_mdlstm3'][:]
        W_ub_mdlstm3=np.concatenate((V_x4_mdlstm3,V_h4_mdlstm3, V_v4_mdlstm3), axis=0)       
        b_ub_mdlstm3=fh5['mdlstm3']['b4_mdlstm3']
        
        
        # 1 AACHEN --> dwd-fwd
        V_x1_mdlstm4, V_h1_mdlstm4, V_v1_mdlstm4=fh5['mdlstm4']['W1_mdlstm4'][:], fh5['mdlstm4']['U1_mdlstm4'][:], fh5['mdlstm4']['V1_mdlstm4'][:]
        W_df_mdlstm4=np.concatenate((V_x1_mdlstm4,V_h1_mdlstm4, V_v1_mdlstm4), axis=0)       
        b_df_mdlstm4=fh5['mdlstm4']['b1_mdlstm4']
        
        # 2 AACHEN --> uwd-fwd
        V_x2_mdlstm4, V_h2_mdlstm4, V_v2_mdlstm4=fh5['mdlstm4']['W2_mdlstm4'][:], fh5['mdlstm4']['U2_mdlstm4'][:], fh5['mdlstm4']['V2_mdlstm4'][:]
        W_uf_mdlstm4=np.concatenate((V_x2_mdlstm4,V_h2_mdlstm4, V_v2_mdlstm4), axis=0)       
        b_uf_mdlstm4=fh5['mdlstm4']['b2_mdlstm4']
        
        # 2 AACHEN --> dwd-bwd
        V_x3_mdlstm4, V_h3_mdlstm4, V_v3_mdlstm4=fh5['mdlstm4']['W3_mdlstm4'][:], fh5['mdlstm4']['U3_mdlstm4'][:], fh5['mdlstm4']['V3_mdlstm4'][:]
        W_db_mdlstm4=np.concatenate((V_x3_mdlstm4,V_h3_mdlstm4, V_v3_mdlstm4), axis=0)       
        b_db_mdlstm4=fh5['mdlstm4']['b3_mdlstm4']
        
        # 2 AACHEN --> uwd-bwd
        V_x4_mdlstm4, V_h4_mdlstm4, V_v4_mdlstm4=fh5['mdlstm4']['W4_mdlstm4'][:], fh5['mdlstm4']['U4_mdlstm4'][:], fh5['mdlstm4']['V4_mdlstm4'][:]
        W_ub_mdlstm4=np.concatenate((V_x4_mdlstm4,V_h4_mdlstm4, V_v4_mdlstm4), axis=0)       
        b_ub_mdlstm4=fh5['mdlstm4']['b4_mdlstm4']
        
        W_dense = fh5['output']['W_in_mdlstm4_output']
        b_dense = fh5['output']['b_output']
        
        return [w_conv0, b_conv0, 
                w_conv1, b_conv1,
                w_conv2, b_conv2,
                w_conv3, b_conv3,
                w_conv4, b_conv4,
                W_df_mdlstm0, b_df_mdlstm0, W_uf_mdlstm0, b_uf_mdlstm0, W_db_mdlstm0, b_db_mdlstm0, W_ub_mdlstm0, b_ub_mdlstm0,
                W_df_mdlstm1, b_df_mdlstm1, W_uf_mdlstm1, b_uf_mdlstm1, W_db_mdlstm1, b_db_mdlstm1, W_ub_mdlstm1, b_ub_mdlstm1,
                W_df_mdlstm2, b_df_mdlstm2, W_uf_mdlstm2, b_uf_mdlstm2, W_db_mdlstm2, b_db_mdlstm2, W_ub_mdlstm2, b_ub_mdlstm2,
                W_df_mdlstm3, b_df_mdlstm3, W_uf_mdlstm3, b_uf_mdlstm3, W_db_mdlstm3, b_db_mdlstm3, W_ub_mdlstm3, b_ub_mdlstm3,
                W_df_mdlstm4, b_df_mdlstm4, W_uf_mdlstm4, b_uf_mdlstm4, W_db_mdlstm4, b_db_mdlstm4, W_ub_mdlstm4, b_ub_mdlstm4,
                W_dense, b_dense]

def get_keys_from_graph(graph):
    """
    It returns specific assign operations from graph in order to assign values from another framework
    to the tensorflow network.
    
    Args:
        graph: a tensorflow graph
        
    Returns:
        The assignation operations of all the parameters in the network.
    """
    
    assign_w_conv0=graph.get_operation_by_name("conv0/conv2d/kernel/Assign")
    assign_b_conv0=graph.get_operation_by_name("conv0/conv2d/bias/Assign")
    w_conv0=assign_w_conv0.inputs[1]   
    b_conv0=assign_b_conv0.inputs[1]
    
    assign_w_conv1=graph.get_operation_by_name("conv1/conv2d/kernel/Assign")
    assign_b_conv1=graph.get_operation_by_name("conv1/conv2d/bias/Assign")
    w_conv1=assign_w_conv1.inputs[1]   
    b_conv1=assign_b_conv1.inputs[1]
    
    assign_w_conv2=graph.get_operation_by_name("conv2/conv2d/kernel/Assign")
    assign_b_conv2=graph.get_operation_by_name("conv2/conv2d/bias/Assign")
    w_conv2=assign_w_conv2.inputs[1]   
    b_conv2=assign_b_conv2.inputs[1]
    
    assign_w_conv3=graph.get_operation_by_name("conv3/conv2d/kernel/Assign")
    assign_b_conv3=graph.get_operation_by_name("conv3/conv2d/bias/Assign")
    w_conv3=assign_w_conv3.inputs[1]   
    b_conv3=assign_b_conv3.inputs[1]
    
    assign_w_conv4=graph.get_operation_by_name("conv4/conv2d/kernel/Assign")
    assign_b_conv4=graph.get_operation_by_name("conv4/conv2d/bias/Assign")
    w_conv4=assign_w_conv4.inputs[1]   
    b_conv4=assign_b_conv4.inputs[1]
    
    
    
    assign_W_df_mdlstm0=graph.get_operation_by_name("MDLSTM_0_dwd-fwd/rnn/kernel/Assign")
    assign_b_df_mdlstm0=graph.get_operation_by_name("MDLSTM_0_dwd-fwd/rnn/bias/Assign")
    W_df_mdlstm0=assign_W_df_mdlstm0.inputs[1]   
    b_df_mdlstm0=assign_b_df_mdlstm0.inputs[1]
    
    assign_W_uf_mdlstm0=graph.get_operation_by_name("MDLSTM_0_uwd-fwd/rnn/kernel/Assign")
    assign_b_uf_mdlstm0=graph.get_operation_by_name("MDLSTM_0_uwd-fwd/rnn/bias/Assign")
    W_uf_mdlstm0=assign_W_uf_mdlstm0.inputs[1]   
    b_uf_mdlstm0=assign_b_uf_mdlstm0.inputs[1]
    
    assign_W_db_mdlstm0=graph.get_operation_by_name("MDLSTM_0_dwd-bwd/rnn/kernel/Assign")
    assign_b_db_mdlstm0=graph.get_operation_by_name("MDLSTM_0_dwd-bwd/rnn/bias/Assign")
    W_db_mdlstm0=assign_W_db_mdlstm0.inputs[1]   
    b_db_mdlstm0=assign_b_db_mdlstm0.inputs[1]
    
    assign_W_ub_mdlstm0=graph.get_operation_by_name("MDLSTM_0_uwd-bwd/rnn/kernel/Assign")
    assign_b_ub_mdlstm0=graph.get_operation_by_name("MDLSTM_0_uwd-bwd/rnn/bias/Assign")
    W_ub_mdlstm0=assign_W_ub_mdlstm0.inputs[1]   
    b_ub_mdlstm0=assign_b_ub_mdlstm0.inputs[1]
    
    
    assign_W_df_mdlstm1=graph.get_operation_by_name("MDLSTM_1_dwd-fwd/rnn/kernel/Assign")
    assign_b_df_mdlstm1=graph.get_operation_by_name("MDLSTM_1_dwd-fwd/rnn/bias/Assign")
    W_df_mdlstm1=assign_W_df_mdlstm1.inputs[1]   
    b_df_mdlstm1=assign_b_df_mdlstm1.inputs[1]
    
    assign_W_uf_mdlstm1=graph.get_operation_by_name("MDLSTM_1_uwd-fwd/rnn/kernel/Assign")
    assign_b_uf_mdlstm1=graph.get_operation_by_name("MDLSTM_1_uwd-fwd/rnn/bias/Assign")
    W_uf_mdlstm1=assign_W_uf_mdlstm1.inputs[1]   
    b_uf_mdlstm1=assign_b_uf_mdlstm1.inputs[1]
    
    assign_W_db_mdlstm1=graph.get_operation_by_name("MDLSTM_1_dwd-bwd/rnn/kernel/Assign")
    assign_b_db_mdlstm1=graph.get_operation_by_name("MDLSTM_1_dwd-bwd/rnn/bias/Assign")
    W_db_mdlstm1=assign_W_db_mdlstm1.inputs[1]   
    b_db_mdlstm1=assign_b_db_mdlstm1.inputs[1]
    
    assign_W_ub_mdlstm1=graph.get_operation_by_name("MDLSTM_1_uwd-bwd/rnn/kernel/Assign")
    assign_b_ub_mdlstm1=graph.get_operation_by_name("MDLSTM_1_uwd-bwd/rnn/bias/Assign")
    W_ub_mdlstm1=assign_W_ub_mdlstm1.inputs[1]   
    b_ub_mdlstm1=assign_b_ub_mdlstm1.inputs[1]
    
    
    
    assign_W_df_mdlstm2=graph.get_operation_by_name("MDLSTM_2_dwd-fwd/rnn/kernel/Assign")
    assign_b_df_mdlstm2=graph.get_operation_by_name("MDLSTM_2_dwd-fwd/rnn/bias/Assign")
    W_df_mdlstm2=assign_W_df_mdlstm2.inputs[1]   
    b_df_mdlstm2=assign_b_df_mdlstm2.inputs[1]
    
    assign_W_uf_mdlstm2=graph.get_operation_by_name("MDLSTM_2_uwd-fwd/rnn/kernel/Assign")
    assign_b_uf_mdlstm2=graph.get_operation_by_name("MDLSTM_2_uwd-fwd/rnn/bias/Assign")
    W_uf_mdlstm2=assign_W_uf_mdlstm2.inputs[1]   
    b_uf_mdlstm2=assign_b_uf_mdlstm2.inputs[1]
    
    assign_W_db_mdlstm2=graph.get_operation_by_name("MDLSTM_2_dwd-bwd/rnn/kernel/Assign")
    assign_b_db_mdlstm2=graph.get_operation_by_name("MDLSTM_2_dwd-bwd/rnn/bias/Assign")
    W_db_mdlstm2=assign_W_db_mdlstm2.inputs[1]   
    b_db_mdlstm2=assign_b_db_mdlstm2.inputs[1]
    
    assign_W_ub_mdlstm2=graph.get_operation_by_name("MDLSTM_2_uwd-bwd/rnn/kernel/Assign")
    assign_b_ub_mdlstm2=graph.get_operation_by_name("MDLSTM_2_uwd-bwd/rnn/bias/Assign")
    W_ub_mdlstm2=assign_W_ub_mdlstm2.inputs[1]   
    b_ub_mdlstm2=assign_b_ub_mdlstm2.inputs[1]
    
    
    
    assign_W_df_mdlstm3=graph.get_operation_by_name("MDLSTM_3_dwd-fwd/rnn/kernel/Assign")
    assign_b_df_mdlstm3=graph.get_operation_by_name("MDLSTM_3_dwd-fwd/rnn/bias/Assign")
    W_df_mdlstm3=assign_W_df_mdlstm3.inputs[1]   
    b_df_mdlstm3=assign_b_df_mdlstm3.inputs[1]
    
    assign_W_uf_mdlstm3=graph.get_operation_by_name("MDLSTM_3_uwd-fwd/rnn/kernel/Assign")
    assign_b_uf_mdlstm3=graph.get_operation_by_name("MDLSTM_3_uwd-fwd/rnn/bias/Assign")
    W_uf_mdlstm3=assign_W_uf_mdlstm3.inputs[1]   
    b_uf_mdlstm3=assign_b_uf_mdlstm3.inputs[1]
    
    assign_W_db_mdlstm3=graph.get_operation_by_name("MDLSTM_3_dwd-bwd/rnn/kernel/Assign")
    assign_b_db_mdlstm3=graph.get_operation_by_name("MDLSTM_3_dwd-bwd/rnn/bias/Assign")
    W_db_mdlstm3=assign_W_db_mdlstm3.inputs[1]   
    b_db_mdlstm3=assign_b_db_mdlstm3.inputs[1]
    
    assign_W_ub_mdlstm3=graph.get_operation_by_name("MDLSTM_3_uwd-bwd/rnn/kernel/Assign")
    assign_b_ub_mdlstm3=graph.get_operation_by_name("MDLSTM_3_uwd-bwd/rnn/bias/Assign")
    W_ub_mdlstm3=assign_W_ub_mdlstm3.inputs[1]   
    b_ub_mdlstm3=assign_b_ub_mdlstm3.inputs[1]
    
    
    
    assign_W_df_mdlstm4=graph.get_operation_by_name("MDLSTM_4_dwd-fwd/rnn/kernel/Assign")
    assign_b_df_mdlstm4=graph.get_operation_by_name("MDLSTM_4_dwd-fwd/rnn/bias/Assign")
    W_df_mdlstm4=assign_W_df_mdlstm4.inputs[1]   
    b_df_mdlstm4=assign_b_df_mdlstm4.inputs[1]
    
    assign_W_uf_mdlstm4=graph.get_operation_by_name("MDLSTM_4_uwd-fwd/rnn/kernel/Assign")
    assign_b_uf_mdlstm4=graph.get_operation_by_name("MDLSTM_4_uwd-fwd/rnn/bias/Assign")
    W_uf_mdlstm4=assign_W_uf_mdlstm4.inputs[1]   
    b_uf_mdlstm4=assign_b_uf_mdlstm4.inputs[1]
    
    assign_W_db_mdlstm4=graph.get_operation_by_name("MDLSTM_4_dwd-bwd/rnn/kernel/Assign")
    assign_b_db_mdlstm4=graph.get_operation_by_name("MDLSTM_4_dwd-bwd/rnn/bias/Assign")
    W_db_mdlstm4=assign_W_db_mdlstm4.inputs[1]   
    b_db_mdlstm4=assign_b_db_mdlstm4.inputs[1]
    
    assign_W_ub_mdlstm4=graph.get_operation_by_name("MDLSTM_4_uwd-bwd/rnn/kernel/Assign")
    assign_b_ub_mdlstm4=graph.get_operation_by_name("MDLSTM_4_uwd-bwd/rnn/bias/Assign")
    W_ub_mdlstm4=assign_W_ub_mdlstm4.inputs[1]   
    b_ub_mdlstm4=assign_b_ub_mdlstm4.inputs[1]
    
    assign_W_dense=graph.get_operation_by_name("Collapse/dense/kernel/Assign")
    assign_b_dense=graph.get_operation_by_name("Collapse/dense/bias/Assign")
    W_dense=assign_W_dense.inputs[1]   
    b_dense=assign_b_dense.inputs[1]
    
    return [w_conv0, b_conv0, 
                w_conv1, b_conv1,
                w_conv2, b_conv2,
                w_conv3, b_conv3,
                w_conv4, b_conv4,
                W_df_mdlstm0, b_df_mdlstm0, W_uf_mdlstm0, b_uf_mdlstm0, W_db_mdlstm0, b_db_mdlstm0, W_ub_mdlstm0, b_ub_mdlstm0,
                W_df_mdlstm1, b_df_mdlstm1, W_uf_mdlstm1, b_uf_mdlstm1, W_db_mdlstm1, b_db_mdlstm1, W_ub_mdlstm1, b_ub_mdlstm1,
                W_df_mdlstm2, b_df_mdlstm2, W_uf_mdlstm2, b_uf_mdlstm2, W_db_mdlstm2, b_db_mdlstm2, W_ub_mdlstm2, b_ub_mdlstm2,
                W_df_mdlstm3, b_df_mdlstm3, W_uf_mdlstm3, b_uf_mdlstm3, W_db_mdlstm3, b_db_mdlstm3, W_ub_mdlstm3, b_ub_mdlstm3,
                W_df_mdlstm4, b_df_mdlstm4, W_uf_mdlstm4, b_uf_mdlstm4, W_db_mdlstm4, b_db_mdlstm4, W_ub_mdlstm4, b_ub_mdlstm4,
                W_dense, b_dense]

def initialize_from_AACHEN(graph, AACHEN_h5_file, log_file_indicator):
    """
    It assign the parameters from the https://github.com/rwth-i6/returnn framework on their mdlstm
    demo network to initialize the parameters over a tensorflow graph.
    
    Args:
        graph:
        AACHEN_h5_file: 
        log_file_indicator:
            
    Returns:
        a python dictionary containing pairs tensorflow parameters - values to initialize the network.
    """
    values = load_AACHEN_PARAMS(AACHEN_h5_file, log_file_indicator)
    keys = get_keys_from_graph(graph)
    
    return dict(zip(keys,values))

def pack_images(image_batch_list, imageHeight, imageWidth):
    """
    It pad all the images in image_batch_list in order to have the same height and
    the same width.
    
    Args:
        image_batch_list: list with the images in their original size.
        imageHeight: the required height for resize all the images.
        imageWidth: the required width for resize all the images.
        
    Returns:
        image_batch_list: list with the resizes images.
            
    """
    import numpy as np
    for ind, img in enumerate(image_batch_list):
        img = np.pad(img,((0,imageHeight-img.shape[0]),(0,imageWidth- img.shape[1])),mode='constant', constant_values=0)
        img = img.reshape(imageHeight, imageWidth, 1)
        image_batch_list[ind] = img
    return image_batch_list
        
def main():
    
    log_path='../Projects/IAM_words/Structure_001/log/log.txt'
    
    h5_file_path='../Projects/IAM_words/Sets/demoset.h5'
    DataBasePath='../Projects/IAM_words/DataBase/words/'
    
    log_file_indicator=initialize_log(log_path)
    
    load_dataset(h5_file_path, DataBasePath, log_file_indicator)

    log_file_indicator.close()



if __name__ == "__main__":
    main()
