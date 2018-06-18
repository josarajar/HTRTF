
import tensorflow as tf
from math import ceil
from MDLSTM import BasicMultidimensionalLSTMCell, multidimensional_dynamic_rnn

def bidirectionalLSTM(inputs, num_hidden, seq_length, name, evalFLAG):
    '''
    It computes 1D-bidirectional LSTM layer in a tensorflow graph extracting some
    features to be shown in tensorboard.
    
    Args:
        inputs: 4D-tensor with the inputs in format (BHWC).
        num_hidden: the number of hidden layers of each layer in both directions.
        seq_length: list with the width of the input images without padding
        name: string with the name of the layer in the graph.
        evalFLAG: boolean indicating if the task is evaluation.
        
    Returns:
        ouputs: tuple with the outputs of the two directions
        
    '''
    with tf.variable_scope('BLSTM'+name) as vs:
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        
        outputs, _= tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, inputs, sequence_length = seq_length, dtype=tf.float32,)
        
        # Retrieve just the LSTM variables.
        
        w_fw, b_fw, w_bw, b_bw = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=vs.name)
        
        with tf.name_scope('lstm_forward'):
            
            with tf.name_scope('input_gate_weights'):
                w_fw_i = tf.split(value=w_fw, num_or_size_splits=4, axis=1, name='input_gate_weights')[0]
                if not evalFLAG:
                    variable_summaries(w_fw_i)
            with tf.name_scope('input_weights'):
                w_fw_j = tf.split(value=w_fw, num_or_size_splits=4, axis=1, name='input_weights')[1]   
                if not evalFLAG:
                    variable_summaries(w_fw_j)
            with tf.name_scope('forget_gate_weights'):
                w_fw_f = tf.split(value=w_fw, num_or_size_splits=4, axis=1, name='forget_gate_weights')[2]
                if not evalFLAG:
                    variable_summaries(w_fw_f)
            with tf.name_scope('output_gate_weights'):
                w_fw_o = tf.split(value=w_fw, num_or_size_splits=4, axis=1, name='output_gate_weights')[3]
                if not evalFLAG:
                    variable_summaries(w_fw_o)
            
            with tf.name_scope('input_gate_biases'):
                b_fw_i = tf.split(value=b_fw, num_or_size_splits=4, axis=0, name='input_gate_biases')[0]
                if not evalFLAG:
                    variable_summaries(b_fw_i)
            with tf.name_scope('input_biases'):
                b_fw_j = tf.split(value=b_fw, num_or_size_splits=4, axis=0, name='input_biases')[1]
                if not evalFLAG:
                    variable_summaries(b_fw_j)
            with tf.name_scope('forget_gate_biases'):
                b_fw_f = tf.split(value=b_fw, num_or_size_splits=4, axis=0, name='forget_gate_biases')[2]
                if not evalFLAG:
                    variable_summaries(b_fw_f)
            with tf.name_scope('output_gate_biases'):
                b_fw_o = tf.split(value=b_fw, num_or_size_splits=4, axis=0, name='output_gate_biases')[3]
                if not evalFLAG:
                    variable_summaries(b_fw_o)
            

        with tf.name_scope('lstm_backward'):
        
            with tf.name_scope('input_gate_weights'):
                w_bw_i = tf.split(value=w_bw, num_or_size_splits=4, axis=1, name='input_gate_weights')[0]
                if not evalFLAG:
                    variable_summaries(w_bw_i)
            with tf.name_scope('input_weights'):
                w_bw_j = tf.split(value=w_bw, num_or_size_splits=4, axis=1, name='input_weights')[1]   
                if not evalFLAG:
                    variable_summaries(w_bw_j)
            with tf.name_scope('forget_gate_weights'):
                w_bw_f = tf.split(value=w_bw, num_or_size_splits=4, axis=1, name='forget_gate_weights')[2]
                if not evalFLAG:
                    variable_summaries(w_bw_f)
            with tf.name_scope('output_gate_weights'):
                w_bw_o = tf.split(value=w_bw, num_or_size_splits=4, axis=1, name='output_gate_weights')[3]
                if not evalFLAG:
                    variable_summaries(w_bw_o)
            
            with tf.name_scope('input_gate_biases'):
                b_bw_i = tf.split(value=b_bw, num_or_size_splits=4, axis=0, name='input_gate_biases')[0]
                if not evalFLAG:
                    variable_summaries(b_bw_i)
            with tf.name_scope('input_biases'):
                b_bw_j = tf.split(value=b_bw, num_or_size_splits=4, axis=0, name='input_biases')[1]
                if not evalFLAG:
                    variable_summaries(b_bw_j)
            with tf.name_scope('forget_gate_biases'):
                b_bw_f = tf.split(value=b_bw, num_or_size_splits=4, axis=0, name='forget_gate_biases')[2]
                if not evalFLAG:
                    variable_summaries(b_bw_f)
            with tf.name_scope('output_gate_biases'):
                b_bw_o = tf.split(value=b_bw, num_or_size_splits=4, axis=0, name='output_gate_biases')[3]
                if not evalFLAG:
                    variable_summaries(b_bw_o)
            
       
        return outputs         

def variable_summaries(var):
  '''
  It saves some scalars so it could be shown in tensorboard.
  
  Args:
      var: tensor of which mean, standard deviation, max, min and histogram will be computed.
      
  '''  
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)
    
def weight_variable(shape, name, collections=None):
    '''
    It initializes a variable tensor of shape `shape` with samples of a normal
    distribution of standar deviation 0.01.
    
    Args:
        shape: shape of the initialized tensor
        name: name of the tensor in the graph
        collections: collections in which this variable will be inclueded.
        
    Returns:
        
        a tensorflow variable initialized.
        
    '''
    initial = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(initial,name=name, collections=[tf.GraphKeys.GLOBAL_VARIABLES, collections])

def FNN(x, units ,name, activation, evalFLAG):
    '''
    It adds a fully connected layer of `units` units in a tensorflow graph and
    applies it to the input tensor `x`. The values of the parameters on each 
    iteration are saved in order to be shown in tensorboard.
    
    Args:
        x: the inputs to the layer (B x F)
        units: the number of units in the fully connected layer.
        name: the identification name in the graph.
        activation: the activation function of the layer.
        evalFLAG: if set to True, values of the parameters won't be saved in
            tensorboard.

    Returns:
        h: the output of the layer.
    '''
    
    with tf.variable_scope(name) as vs:        
        h=tf.layers.dense(inputs=x, units=units, activation=activation, kernel_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01), bias_initializer=tf.zeros_initializer())
        weights, biases = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=vs.name)
      
        if not evalFLAG:
            with tf.name_scope('weights'):
                variable_summaries(weights)
            with tf.name_scope('biases'):
                variable_summaries(biases)   
    return h

def MultiDirectionalFNN(x_df, x_uf, x_db, x_ub, units, name, activation, evalFLAG):
    '''
    It applies four fully connected layers in parallel at the output of a
    MDLSTM in which the input image is sweept in four directions from each
    corner. After that, it also sumes the four outputs.
    
    Args:
        x_df: the inputs to the layer in the downward-forward direction (B x F)
        x_uf: the inputs to the layer in the upward-forward direction (B x F)
        x_db: the inputs to the layer in the downward-backward direction (B x F)
        x_ub: the inputs to the layer in the upward-backward direction (B x F)
        untis:the number of units in each layer
        name: the identification name in the graph.
        activaition: the activation function of the layer.
        evalFLAG: if set to True, values of the parameters won't be saved in
            tensorboard.
            
    Returns:
        h: the output of the layer.
        
    '''
    
    _ , imageHeight, imageWidth, channels =x_df.get_shape().as_list()
    
    h_df = tf.reshape(x_df,[-1, channels])
    h_uf = tf.reshape(x_uf,[-1, channels])
    h_db = tf.reshape(x_db,[-1, channels])
    h_ub = tf.reshape(x_ub,[-1, channels])
    
    h_df = FNN(h_df, units, name=name+'downward-forward', activation=activation, evalFLAG=evalFLAG) 
    h_uf = FNN(h_uf, units, name=name+'upward-forward', activation=activation, evalFLAG=evalFLAG) 
    h_db = FNN(h_db, units, name=name+'downward-backward', activation=activation, evalFLAG=evalFLAG) 
    h_ub = FNN(h_ub, units, name=name+'upward-backward', activation=activation, evalFLAG=evalFLAG)       
    
    h=h_df+h_uf+h_db+h_ub
    
    h=tf.reshape(h, [-1,imageHeight, imageWidth, units]) 
    
    return h
        

def CNN(x, filters, kernel_size, strides, name, activation, evalFLAG, initializer=tf.contrib.layers.xavier_initializer()):
    '''
    It adds a convolutional layer of `filters` number of filters in a tensorflow graph and
    applies it to the input tensor `x`. The values of the parameters on each 
    iteration are saved in order to be shown in tensorboard.
    
    Args:
        x: the inputs to the layer (B x F)
        filters: the number of filters in the convolutional layer.
        kernel_size: the size of the kernel.
        strides: the stride of the kernel
        name: the name of the layer in the graph.
        activation: the activation function of the layer.
        evalFLAG: if set to True, values of the parameters won't be saved in
            tensorboard.
        initializer: the initializer of the values of the parameters.

    Returns:
        h: the output of the layer.
    '''

    with tf.variable_scope(name) as vs:
        h=tf.layers.conv2d(inputs=x, filters=filters, kernel_size=kernel_size, padding='same', activation=activation, kernel_initializer=initializer, bias_initializer=tf.zeros_initializer())
        weights, biases = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=vs.name)
        if not evalFLAG:
                with tf.name_scope(name+'-weights'):
                    variable_summaries(weights)
                with tf.name_scope(name+'-biases'):
                    variable_summaries(biases)   
        else:
            for ind in range(filters):    
                tf.summary.image('Conv_activations',h[5:6,:,:,ind:ind+1],max_outputs=1)
    
    return h

def MultiDirectionalCNN(x_df, x_uf, x_db, x_ub, filters, kernel_size, strides, name, activation, evalFLAG):
    '''
    It applies four convolutional layers in parallel at the output of a
    MDLSTM in which the input image is sweept in four directions from each
    corner. 
    
    Args:
        x_df: the inputs to the layer in the downward-forward direction (B x F)
        x_uf: the inputs to the layer in the upward-forward direction (B x F)
        x_db: the inputs to the layer in the downward-backward direction (B x F)
        x_ub: the inputs to the layer in the upward-backward direction (B x F)
        filters: the number of filters in the convolutional layer.
        kernel_size: the size of the kernel.
        strides: the stride of the kernel
        name: the identification name in the graph.
        activaition: the activation function of the layer.
        evalFLAG: if set to True, values of the parameters won't be saved in
            tensorboard.
            
    Returns:
        h: the output of the layer.
        
    '''
    h_conv_df = CNN(x_df, filters, kernel_size, strides, name=name+'downward-forward', activation=activation, evalFLAG=evalFLAG) 
    h_conv_uf = CNN(x_uf, filters, kernel_size, strides, name=name+'upward-forward', activation=activation, evalFLAG=evalFLAG) 
    h_conv_db = CNN(x_db, filters, kernel_size, strides, name=name+'downward-backward', activation=activation, evalFLAG=evalFLAG) 
    h_conv_ub = CNN(x_ub, filters, kernel_size, strides, name=name+'upward-backward', activation=activation, evalFLAG=evalFLAG) 
     
    h_conv=tf.tanh(h_conv_df+h_conv_uf+h_conv_db+h_conv_ub)
    
    return h_conv

def max_pool(x, pool_size, seq_len, imageHeight, imageWidth, evalFLAG):
    h = tf.nn.max_pool(x, ksize=[1,pool_size[0],pool_size[1],1], strides=[1,pool_size[0],pool_size[1],1], padding='SAME')
    seq_len = tf.cast(tf.ceil(seq_len/pool_size[1]),dtype=tf.int32) 
    imageHeight = ceil(imageHeight/pool_size[0])
    imageWidth = ceil(imageWidth/pool_size[1])
    if evalFLAG:
        tf.summary.image('Pool1',h[:,:,:,5:6],max_outputs=1)
    return h, seq_len, imageHeight, imageWidth

def conv2d(x, W):
    return tf.nn.conv2d(x,W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x, seq_len, imageHeight, imageWidth):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME'), tf.cast(tf.ceil(seq_len/2),dtype=tf.int32), ceil(imageHeight/2), ceil(imageWidth/2)


def ln(tensor, scope=None, epsilon=1e-5):
    """ Layer normalizes a 2D tensor along its second axis """
    assert (len(tensor.get_shape()) == 2)
    m, v = tf.nn.moments(tensor, [1], keep_dims=True)
    if not isinstance(scope, str):
        scope = ''
    with tf.variable_scope(scope + 'layer_norm'):
        scale = tf.get_variable('scale',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(1))
        shift = tf.get_variable('shift',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(0))
    ln_initial = (tensor - m) / tf.sqrt(v + epsilon)

    return ln_initial * scale + shift



def extract_2x2_patches(x,seq_len, imageHeight, imageWidth):
    output=tf.extract_image_patches(x,ksizes=[1, 2, 2, 1], strides=[1, 2, 2, 1], rates=[1, 1, 1, 1], padding='SAME')
    seq_len = tf.cast(tf.ceil(seq_len/2),dtype=tf.int32) 
    imageHeight = ceil(imageHeight/2)
    imageWidth = ceil(imageWidth/2)
    
    return output, seq_len, imageHeight, imageWidth



def real_basic_mdlstm(input_data, rnn_size, dims=None, scope_n='dwd-fwd'):
    
    with tf.variable_scope(scope_n):
        if dims is not None:
            assert 0 not in dims and 3 not in dims
            input_data=tf.reverse(input_data,dims)
            
        _ , h, w, c = input_data.get_shape().as_list()
        cell = BasicMultidimensionalLSTMCell(rnn_size)
        
        rnn_out, _ = multidimensional_dynamic_rnn(cell, input_data, h, w, dtype=tf.float32)
        
        
        if dims is not None:
            rnn_out=tf.reverse(rnn_out,dims)
            
    return rnn_out

def BasicMultiDirectionalMultidimensionalLSTM(x, rnn_size, imageWidth, chunkSize, name, evalFLAG=False):
                
    #Chunking
#    if not evalFLAG:
    x=tf.pad(x, [[0,0],[0,0],[0,ceil(imageWidth/chunkSize)*chunkSize-imageWidth],[0,0]])
    x=tf.concat(tf.split(x,ceil(imageWidth/chunkSize),axis=2),axis=0)    
    
    h_mdlstm_df = real_basic_mdlstm(input_data=x, rnn_size=rnn_size, scope_n=name+'_dwd-fwd')
    h_mdlstm_uf = real_basic_mdlstm(input_data=x, rnn_size=rnn_size, dims=[1], scope_n=name+'_uwd-fwd')
    h_mdlstm_db = real_basic_mdlstm(input_data=x, rnn_size=rnn_size, dims=[2], scope_n=name+'_dwd-bwd')
    h_mdlstm_ub = real_basic_mdlstm(input_data=x, rnn_size=rnn_size, dims=[1,2], scope_n=name+'_uwd-bwd')
    
    #Undo chunking
#    if not evalFLAG:
    h_mdlstm_df=tf.concat(tf.split(h_mdlstm_df,ceil(imageWidth/chunkSize),axis=0),axis=2)
    h_mdlstm_uf=tf.concat(tf.split(h_mdlstm_uf,ceil(imageWidth/chunkSize),axis=0),axis=2)
    h_mdlstm_db=tf.concat(tf.split(h_mdlstm_db,ceil(imageWidth/chunkSize),axis=0),axis=2)
    h_mdlstm_ub=tf.concat(tf.split(h_mdlstm_ub,ceil(imageWidth/chunkSize),axis=0),axis=2)

    h_mdlstm_df=tf.slice(h_mdlstm_df,[0,0,0,0],[-1,-1,imageWidth,-1])
    h_mdlstm_uf=tf.slice(h_mdlstm_uf,[0,0,0,0],[-1,-1,imageWidth,-1])
    h_mdlstm_db=tf.slice(h_mdlstm_db,[0,0,0,0],[-1,-1,imageWidth,-1])
    h_mdlstm_ub=tf.slice(h_mdlstm_ub,[0,0,0,0],[-1,-1,imageWidth,-1])

    return h_mdlstm_df, h_mdlstm_uf, h_mdlstm_db, h_mdlstm_ub

def MultiDirectionalDropOut(h_df, h_uf, h_db, h_ub, keep_prob):
    h_df = tf.nn.dropout(h_df,keep_prob=keep_prob)
    h_uf = tf.nn.dropout(h_uf,keep_prob=keep_prob)
    h_db = tf.nn.dropout(h_db,keep_prob=keep_prob)
    h_ub = tf.nn.dropout(h_ub,keep_prob=keep_prob)
    
    return h_df, h_uf, h_db, h_ub
    
