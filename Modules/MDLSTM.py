from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import constant_op

Initializer = init_ops.Initializer




import collections

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

# pylint: disable=protected-access
def _concat(prefix, suffix, static=False):
  """Concat that enables int, Tensor, or TensorShape values.
  This function takes a size specification, which can be an integer, a
  TensorShape, or a Tensor, and converts it into a concatenated Tensor
  (if static = False) or a list of integers (if static = True).
  Args:
    prefix: The prefix; usually the batch size (and/or time step size).
      (TensorShape, int, or Tensor.)
    suffix: TensorShape, int, or Tensor.
    static: If `True`, return a python list with possibly unknown dimensions.
      Otherwise return a `Tensor`.
  Returns:
    shape: the concatenation of prefix and suffix.
  Raises:
    ValueError: if `suffix` is not a scalar or vector (or TensorShape).
    ValueError: if prefix or suffix was `None` and asked for dynamic
      Tensors out.
  """
  if isinstance(prefix, ops.Tensor):
    p = prefix
    p_static = tensor_util.constant_value(prefix)
    if p.shape.ndims == 0:
      p = array_ops.expand_dims(p, 0)
    elif p.shape.ndims != 1:
      raise ValueError("prefix tensor must be either a scalar or vector, "
                       "but saw tensor: %s" % p)
  else:
    p = tensor_shape.as_shape(prefix)
    p_static = p.as_list() if p.ndims is not None else None
    p = (constant_op.constant(p.as_list(), dtype=dtypes.int32)
         if p.is_fully_defined() else None)
  if isinstance(suffix, ops.Tensor):
    s = suffix
    s_static = tensor_util.constant_value(suffix)
    if s.shape.ndims == 0:
      s = array_ops.expand_dims(s, 0)
    elif s.shape.ndims != 1:
      raise ValueError("suffix tensor must be either a scalar or vector, "
                       "but saw tensor: %s" % s)
  else:
    s = tensor_shape.as_shape(suffix)
    s_static = s.as_list() if s.ndims is not None else None
    s = (constant_op.constant(s.as_list(), dtype=dtypes.int32)
         if s.is_fully_defined() else None)

  if static:
    shape = tensor_shape.as_shape(p_static).concatenate(s_static)
    shape = shape.as_list() if shape.ndims is not None else None
  else:
    if p is None or s is None:
      raise ValueError("Provided a prefix or suffix of None: %s and %s"
                       % (prefix, suffix))
    shape = array_ops.concat((p, s), 0)
  return shape

def _like_rnncell(cell):
  """Checks that a given object is an RNNCell by using duck typing."""
  conditions = [hasattr(cell, "output_size"), hasattr(cell, "state_size"),
                hasattr(cell, "zero_state"), callable(cell)]
  return all(conditions)


_MDLSTMStateTuple = collections.namedtuple("MDLSTMStateTuple", ("c_left","c_top", "h_left", "h_top"))

class MDLSTMStateTuple(_MDLSTMStateTuple):
  """Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.
  Stores two elements: `(c, h)`, in that order. Where `c` is the hidden state
  and `h` is the output.
  Only used when `state_is_tuple=True`.
  """
  __slots__ = ()

  @property
  def dtype(self):
    (c_left, c_top, h_left, h_top) = self
    if c_left.dtype != h_left.dtype or h_left.dtype != c_top.dtype or c_top.dtype != h_top.dtype:
      raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(c_left.dtype), str(h_left.dtype)))
    return c_left.dtype

class ConstantMDLSTMbias(Initializer):

  def __init__(self, dtype=dtypes.float32, verify_shape=False):
    self.dtype = dtypes.as_dtype(dtype)
    self._verify_shape = verify_shape

  def __call__(self, shape, dtype=None, partition_info=None, verify_shape=None):
    if dtype is None:
      dtype = self.dtype
    if verify_shape is None:
      verify_shape = self._verify_shape
    aux_shape=shape[0]/5
    
    return array_ops.concat([constant_op.constant(
        0, dtype=dtype, shape=[aux_shape]), constant_op.constant(
        1, dtype=dtype, shape=[aux_shape]), constant_op.constant(
        0, dtype=dtype, shape=[3*aux_shape])],axis=0)
      
MDLSTM_bias_initializer=ConstantMDLSTMbias
    
class _LinearStable(object):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of weight variable.
    dtype: data type for variables.
    build_bias: boolean, whether to build a bias variable.
    bias_initializer: starting value to initialize the bias
      (default is all zeros).
    kernel_initializer: starting value to initialize the weight.
  Raises:
    ValueError: if inputs_shape is wrong.
  """

  def __init__(self,
               args,
               output_size,
               build_bias,
               bias_initializer=None,
               kernel_initializer=None):
    self._build_bias = build_bias

    if args is None or (nest.is_sequence(args) and not args):
      raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
      args = [args]
      self._is_sequence = False
    else:
      self._is_sequence = True


    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
      if shape.ndims != 2:
        raise ValueError("linear is expecting 2D arguments: %s" % shapes)
      if shape[1].value is None:
        raise ValueError("linear expects shape[1] to be provided for shape %s, "
                         "but saw %s" % (shape, shape[1]))
      else:
        total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
      self._weights = vs.get_variable(
          _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
          dtype=dtype,
          initializer=kernel_initializer)
      if build_bias:
        with vs.variable_scope(outer_scope) as inner_scope:
          inner_scope.set_partitioner(None)
          if bias_initializer is None:
            bias_initializer = MDLSTM_bias_initializer()
          self._biases = vs.get_variable(
              _BIAS_VARIABLE_NAME, [output_size],
              dtype=dtype,
              initializer=bias_initializer)

  def __call__(self, args):
    if not self._is_sequence:
      args = [args]

    if len(args) == 1:
      res = math_ops.matmul(args[0], self._weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), self._weights)
    if self._build_bias:
      res = nn_ops.bias_add(res, self._biases)
    return res



def delay_state(state, width, num_units):
    state_batched = array_ops.reshape(state,[-1, width, num_units])
    state_valid = array_ops.slice(state_batched,[0,0,0],[-1,width-1,-1])
    state_padded = array_ops.pad(state_valid,[[0,0],[1,0],[0,0]])
    new_state = array_ops.reshape(state_padded, [-1, num_units])
    
    return new_state

def delete_padded_gradient(state, height, width, num_units, step):
    import tensorflow as tf
    state_batched = tf.reshape(state,[-1, width, num_units])
    state_valid = tf.slice(state_batched,[0,tf.reduce_max([0,step+1-height]),0],[-1,tf.reduce_min([step+1,width,height+width-1-step, height]),-1])
    state_padded = tf.pad(state_valid,[[0,0],[tf.reduce_max([0,step+1-height]),tf.reduce_max([0,width-(step+1)])],[0,0]])
    new_state = tf.reshape(state_padded, [-1, num_units])
    
    return new_state
  
class BasicMultidimensionalLSTMCell(RNNCell):
  """Basic LSTM recurrent network cell.
  The implementation is based on: http://arxiv.org/abs/1409.2329.
  We add forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.
  It does not allow cell clipping, a projection layer, and does not
  use peep-hole connections: it is the basic baseline.
  For advanced models, please use the full @{tf.nn.rnn_cell.LSTMCell}
  that follows.
  """

  def __init__(self, num_units, 
               state_is_tuple=True, activation=None, reuse=None):
    """Initialize the basic LSTM cell.
    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
        Must set to `0.0` manually when restoring from CudnnLSTM-trained
        checkpoints.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  The latter behavior will soon be deprecated.
      activation: Activation function of the inner states.  Default: `tanh`.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
      When restoring from CudnnLSTM-trained checkpoints, must use
      CudnnCompatibleLSTMCell instead.
    """
    super(BasicMultidimensionalLSTMCell, self).__init__()
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    self._num_units = num_units
    self._state_is_tuple = state_is_tuple
    self._activation = activation or math_ops.tanh
    self._linear = None

  @property
  def state_size(self):
    return (MDLSTMStateTuple(self._num_units, self._num_units, self._num_units, self._num_units)
            if self._state_is_tuple else 4 * self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def call(self, inputs, state, height, width, step):
    """Long short-term memory cell (LSTM).
    Args:
      inputs: `2-D` tensor with shape `[batch_size x input_size]`.
      state: An `LSTMStateTuple` of state tensors, each shaped
        `[batch_size x self.state_size]`, if `state_is_tuple` has been set to
        `True`.  Otherwise, a `Tensor` shaped
        `[batch_size x 4 * self.state_size]`.
    Returns:
      A pair containing the new hidden state, and the new state (either a
        `LSTMStateTuple` or a concatenated state, depending on
        `state_is_tuple`).
    """
    sigmoid = math_ops.sigmoid
    # Parameters of gates are concatenated into one multiply for efficiency.
    if self._state_is_tuple:
      c_left, c_top, h_left, h_top = state
    else:
      c_left, c_top, h_left, h_top = array_ops.split(value=state, num_or_size_splits=4, axis=1)

    if self._linear is None:
      self._linear = _LinearStable([inputs, h_left, h_top], 5 * self._num_units, build_bias=True)
    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    i, f, _lambda, o, j = array_ops.split(
        value=self._linear([inputs, h_left, h_top]), num_or_size_splits=5, axis=1)

    new_c = (
        (c_top * sigmoid(_lambda) + c_left * (1-sigmoid(_lambda)))* sigmoid(f) + sigmoid(i) * self._activation(j))
  
    new_h = self._activation(new_c) * sigmoid(o)
    
    # Delete padding for avoid memory allocation for 0 valued gradients
    new_c = delete_padded_gradient(new_c, height, width, self._num_units, step)
    new_h = delete_padded_gradient(new_h, height, width, self._num_units, step)
    
    # c_top --> c_left, new_c --> c_top, h_top --> h_left, new_h --> h_top
    new_c_left=delay_state(new_c, width, self._num_units)
    new_h_left=delay_state(new_h, width, self._num_units)
    
    if self._state_is_tuple:
      new_state = MDLSTMStateTuple(new_c_left, new_c, new_h_left, new_h)
    else:
      new_state = array_ops.concat([new_c_left, new_c, new_h_left, new_h], 1)
    return new_h, new_state


def _transpose_batch_time(x):
  """Transpose the batch and time dimensions of a Tensor.
  Retains as much of the static shape information as possible.
  Args:
    x: A tensor of rank 2 or higher.
  Returns:
    x transposed along the first two dimensions.
  Raises:
    ValueError: if `x` is rank 1 or lower.
  """
  x_static_shape = x.get_shape()
  if x_static_shape.ndims is not None and x_static_shape.ndims < 2:
    raise ValueError(
        "Expected input tensor %s to have rank at least 2, but saw shape: %s" %
        (x, x_static_shape))
  x_rank = array_ops.rank(x)
  x_t = array_ops.transpose(
      x, array_ops.concat(
          ([1, 0], math_ops.range(2, x_rank)), axis=0))
  x_t.set_shape(
      tensor_shape.TensorShape([
          x_static_shape[1].value, x_static_shape[0].value
      ]).concatenate(x_static_shape[2:]))
  return x_t

def _best_effort_input_batch_size(flat_input):
  """Get static input batch size if available, with fallback to the dynamic one.
  Args:
    flat_input: An iterable of time major input Tensors of shape [max_time,
      batch_size, ...]. All inputs should have compatible batch sizes.
  Returns:
    The batch size in Python integer if available, or a scalar Tensor otherwise.
  Raises:
    ValueError: if there is any input with an invalid shape.
  """
  for input_ in flat_input:
    shape = input_.shape
    if shape.ndims is None:
      continue
    if shape.ndims < 2:
      raise ValueError(
          "Expected input tensor %s to have rank at least 2" % input_)
    batch_size = shape[1].value
    if batch_size is not None:
      return batch_size
  # Fallback to the dynamic batch size of the first input.
  return array_ops.shape(flat_input[0])[1]

def _infer_state_dtype(explicit_dtype, state):
  """Infer the dtype of an RNN state.
  Args:
    explicit_dtype: explicitly declared dtype or None.
    state: RNN's hidden state. Must be a Tensor or a nested iterable containing
      Tensors.
  Returns:
    dtype: inferred dtype of hidden state.
  Raises:
    ValueError: if `state` has heterogeneous dtypes or is empty.
  """
  if explicit_dtype is not None:
    return explicit_dtype
  elif nest.is_sequence(state):
    inferred_dtypes = [element.dtype for element in nest.flatten(state)]
    if not inferred_dtypes:
      raise ValueError("Unable to infer dtype from empty state.")
    all_same = all([x == inferred_dtypes[0] for x in inferred_dtypes])
    if not all_same:
      raise ValueError(
          "State has tensors of different inferred_dtypes. Unable to infer a "
          "single representative dtype.")
    return inferred_dtypes[0]
  else:
    return state.dtype

def multidimensional_dynamic_rnn(cell, inputs, height, width, sequence_length=None, initial_state=None,
                dtype=None, parallel_iterations=None, swap_memory=False, scope=None):
  """Creates a recurrent neural network specified by RNNCell `cell`.
  Performs fully dynamic unrolling of `inputs`.
  Example:
  ```python
  # create a BasicRNNCell
  rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
  # 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
  # defining initial state
  initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
  # 'state' is a tensor of shape [batch_size, cell_state_size]
  outputs, state = tf.nn.dynamic_rnn(rnn_cell, input_data,
                                     initial_state=initial_state,
                                     dtype=tf.float32)
  ```
  ```python
  # create 2 LSTMCells
  rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [128, 256]]
  # create a RNN cell composed sequentially of a number of RNNCells
  multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
  # 'outputs' is a tensor of shape [batch_size, max_time, 256]
  # 'state' is a N-tuple where N is the number of LSTMCells containing a
  # tf.contrib.rnn.LSTMStateTuple for each cell
  outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                     inputs=data,
                                     dtype=tf.float32)
  ```
  Args:
    cell: An instance of RNNCell.
    inputs: The RNN inputs.
      If `time_major == False` (default), this must be a `Tensor` of shape:
        `[batch_size, max_time, ...]`, or a nested tuple of such
        elements.
      If `time_major == True`, this must be a `Tensor` of shape:
        `[max_time, batch_size, ...]`, or a nested tuple of such
        elements.
      This may also be a (possibly nested) tuple of Tensors satisfying
      this property.  The first two dimensions must match across all the inputs,
      but otherwise the ranks and other shape components may differ.
      In this case, input to `cell` at each time-step will replicate the
      structure of these tuples, except for the time dimension (from which the
      time is taken).
      The input to `cell` at each time step will be a `Tensor` or (possibly
      nested) tuple of Tensors each with dimensions `[batch_size, ...]`.
    sequence_length: (optional) An int32/int64 vector sized `[batch_size]`.
      Used to copy-through state and zero-out outputs when past a batch
      element's sequence length.  So it's more for correctness than performance.
    initial_state: (optional) An initial state for the RNN.
      If `cell.state_size` is an integer, this must be
      a `Tensor` of appropriate type and shape `[batch_size, cell.state_size]`.
      If `cell.state_size` is a tuple, this should be a tuple of
      tensors having shapes `[batch_size, s] for s in cell.state_size`.
    dtype: (optional) The data type for the initial state and expected output.
      Required if initial_state is not provided or RNN state has a heterogeneous
      dtype.
    parallel_iterations: (Default: 32).  The number of iterations to run in
      parallel.  Those operations which do not have any temporal dependency
      and can be run in parallel, will be.  This parameter trades off
      time for space.  Values >> 1 use more memory but take less time,
      while smaller values use less memory but computations take longer.
    swap_memory: Transparently swap the tensors produced in forward inference
      but needed for back prop from GPU to CPU.  This allows training RNNs
      which would typically not fit on a single GPU, with very minimal (or no)
      performance penalty.
    time_major: The shape format of the `inputs` and `outputs` Tensors.
      If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
      If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
      Using `time_major = True` is a bit more efficient because it avoids
      transposes at the beginning and end of the RNN calculation.  However,
      most TensorFlow data is batch-major, so by default this function
      accepts input and emits output in batch-major form.
    scope: VariableScope for the created subgraph; defaults to "rnn".
  Returns:
    A pair (outputs, state) where:
    outputs: The RNN output `Tensor`.
      If time_major == False (default), this will be a `Tensor` shaped:
        `[batch_size, max_time, cell.output_size]`.
      If time_major == True, this will be a `Tensor` shaped:
        `[max_time, batch_size, cell.output_size]`.
      Note, if `cell.output_size` is a (possibly nested) tuple of integers
      or `TensorShape` objects, then `outputs` will be a tuple having the
      same structure as `cell.output_size`, containing Tensors having shapes
      corresponding to the shape data in `cell.output_size`.
    state: The final state.  If `cell.state_size` is an int, this
      will be shaped `[batch_size, cell.state_size]`.  If it is a
      `TensorShape`, this will be shaped `[batch_size] + cell.state_size`.
      If it is a (possibly nested) tuple of ints or `TensorShape`, this will
      be a tuple having the corresponding shapes. If cells are `LSTMCells`
      `state` will be a tuple containing a `LSTMStateTuple` for each cell.
  Raises:
    TypeError: If `cell` is not an instance of RNNCell.
    ValueError: If inputs is None or an empty list.
  """
  if not _like_rnncell(cell):
    raise TypeError("cell must be an instance of RNNCell")

  # By default, time_major==False and inputs are batch-major: shaped
  #   [batch, time, depth]
  # For internal calculations, we transpose to [time, batch, depth]
  flat_input = nest.flatten(inputs)
  
  height, width, channels = flat_input[0].get_shape().as_list()[1:]
  
  height, width, channels = flat_input[0].get_shape().as_list()[1:]
  i_unpck=[array_ops.unstack(input_,axis=2) for input_ in flat_input]
  for input_ in i_unpck:
    for column in range(width):
        input_[column]=array_ops.pad(input_[column],[[0,0],[column,width-(1+column)],[0,0]])        
  i_padded = [array_ops.stack(input_,axis=2) for input_ in i_unpck]
  i_bwhc = [array_ops.transpose(input_,[0,2,1,3]) for input_ in i_padded]
  height_expanded=i_bwhc[0].get_shape().as_list()[2]
  i_btc = [array_ops.reshape(input_,[-1,height_expanded,channels]) for input_ in i_bwhc]
  flat_input=[array_ops.transpose(input_, [1,0,2]) for input_ in i_btc]
  
#  if not time_major:
#    # (B,T,D) => (T,B,D)
#    flat_input = [ops.convert_to_tensor(input_) for input_ in flat_input]
#    flat_input = tuple(_transpose_batch_time(input_) for input_ in flat_input)

  parallel_iterations = parallel_iterations or 32
  if sequence_length is not None:
    sequence_length = math_ops.to_int32(sequence_length)
    if sequence_length.get_shape().ndims not in (None, 1):
      raise ValueError(
          "sequence_length must be a vector of length batch_size, "
          "but saw shape: %s" % sequence_length.get_shape())
    sequence_length = array_ops.identity(  # Just to find it in the graph.
        sequence_length, name="sequence_length")

  # Create a new scope in which the caching device is either
  # determined by the parent scope, or is set to place the cached
  # Variable using the same placement as for the rest of the RNN.
  with vs.variable_scope(scope or "rnn") as varscope:
    if varscope.caching_device is None:
      varscope.set_caching_device(lambda op: op.device)
    batch_size = _best_effort_input_batch_size(flat_input)

    if initial_state is not None:
      state = initial_state
    else:
      if not dtype:
        raise ValueError("If there is no initial_state, you must give a dtype.")
      state = cell.zero_state(batch_size, dtype)

    def _assert_has_shape(x, shape):
      x_shape = array_ops.shape(x)
      packed_shape = array_ops.stack(shape)
      return control_flow_ops.Assert(
          math_ops.reduce_all(math_ops.equal(x_shape, packed_shape)),
          ["Expected shape for Tensor %s is " % x.name,
           packed_shape, " but saw shape: ", x_shape])

    if sequence_length is not None:
      # Perform some shape validation
      with ops.control_dependencies(
          [_assert_has_shape(sequence_length, [batch_size])]):
        sequence_length = array_ops.identity(
            sequence_length, name="CheckSeqLen")

    inputs = nest.pack_sequence_as(structure=inputs, flat_sequence=flat_input)

    (outputs, final_state) = _multidimensional_dynamic_rnn_loop(
        cell,
        inputs,
        state,
        height,
        width,
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory,
        sequence_length=sequence_length,
        dtype=dtype)

    # Outputs of _dynamic_rnn_loop are always shaped [time, batch, depth].
    # If we are performing batch-major calculations, transpose output back
    # to shape [batch, time, depth]
#    if not time_major:
#      # (T,B,D) => (B,T,D)
    outputs = nest.map_structure(_transpose_batch_time, outputs)
    
    ch_out=outputs.get_shape().as_list()[2]
#    o_btc=array_ops.transpose(outputs[1,0,2])
    o_bwhc = array_ops.reshape(outputs,[-1,width,height_expanded, ch_out])
    o_padded = array_ops.transpose(o_bwhc, [0,2,1,3])
    o_unpck=array_ops.unstack(o_padded, axis=2)
    for column in range(width):
        o_unpck[column]=array_ops.slice(o_unpck[column],[0,column,0],[-1,height,-1])
    outputs=array_ops.stack(o_unpck,axis=2)

    return (outputs, final_state)

def _multidimensional_dynamic_rnn_loop(cell,
                      inputs,
                      initial_state,
                      height,
                      width,
                      parallel_iterations,
                      swap_memory,
                      sequence_length=None,
                      dtype=None):
  """Internal implementation of Dynamic RNN.
  Args:
    cell: An instance of RNNCell.
    inputs: A `Tensor` of shape [time, batch_size, input_size], or a nested
      tuple of such elements.
    initial_state: A `Tensor` of shape `[batch_size, state_size]`, or if
      `cell.state_size` is a tuple, then this should be a tuple of
      tensors having shapes `[batch_size, s] for s in cell.state_size`.
    parallel_iterations: Positive Python int.
    swap_memory: A Python boolean
    sequence_length: (optional) An `int32` `Tensor` of shape [batch_size].
    dtype: (optional) Expected dtype of output. If not specified, inferred from
      initial_state.
  Returns:
    Tuple `(final_outputs, final_state)`.
    final_outputs:
      A `Tensor` of shape `[time, batch_size, cell.output_size]`.  If
      `cell.output_size` is a (possibly nested) tuple of ints or `TensorShape`
      objects, then this returns a (possibly nsted) tuple of Tensors matching
      the corresponding shapes.
    final_state:
      A `Tensor`, or possibly nested tuple of Tensors, matching in length
      and shapes to `initial_state`.
  Raises:
    ValueError: If the input depth cannot be inferred via shape inference
      from the inputs.
  """
  state = initial_state
  assert isinstance(parallel_iterations, int), "parallel_iterations must be int"

  state_size = cell.state_size

  flat_input = nest.flatten(inputs)
  flat_output_size = nest.flatten(cell.output_size)

  # Construct an initial output
  input_shape = array_ops.shape(flat_input[0])
  time_steps = input_shape[0]
  batch_size = _best_effort_input_batch_size(flat_input)

  inputs_got_shape = tuple(input_.get_shape().with_rank_at_least(3)
                           for input_ in flat_input)

  const_time_steps, const_batch_size = inputs_got_shape[0].as_list()[:2]

  for shape in inputs_got_shape:
    if not shape[2:].is_fully_defined():
      raise ValueError(
          "Input size (depth of inputs) must be accessible via shape inference,"
          " but saw value None.")
    got_time_steps = shape[0].value
    got_batch_size = shape[1].value
    if const_time_steps != got_time_steps:
      raise ValueError(
          "Time steps is not the same for all the elements in the input in a "
          "batch.")
    if const_batch_size != got_batch_size:
      raise ValueError(
          "Batch_size is not the same for all the elements in the input.")

  # Prepare dynamic conditional copying of state & output
  def _create_zero_arrays(size):
    size = _concat(batch_size, size)
    return array_ops.zeros(
        array_ops.stack(size), _infer_state_dtype(dtype, state))

  flat_zero_output = tuple(_create_zero_arrays(output)
                           for output in flat_output_size)
  zero_output = nest.pack_sequence_as(structure=cell.output_size,
                                      flat_sequence=flat_zero_output)

  if sequence_length is not None:
    min_sequence_length = math_ops.reduce_min(sequence_length)
    max_sequence_length = math_ops.reduce_max(sequence_length)

  time = array_ops.constant(0, dtype=dtypes.int32, name="time")

  with ops.name_scope("dynamic_rnn") as scope:
    base_name = scope

  def _create_ta(name, dtype):
    return tensor_array_ops.TensorArray(dtype=dtype,
                                        size=time_steps,
                                        tensor_array_name=base_name + name)

  output_ta = tuple(_create_ta("output_%d" % i,
                               _infer_state_dtype(dtype, state))
                    for i in range(len(flat_output_size)))
  input_ta = tuple(_create_ta("input_%d" % i, flat_input[i].dtype)
                   for i in range(len(flat_input)))

  input_ta = tuple(ta.unstack(input_)
                   for ta, input_ in zip(input_ta, flat_input))

  def _time_step(time, output_ta_t, state):
    """Take a time step of the dynamic RNN.
    Args:
      time: int32 scalar Tensor.
      output_ta_t: List of `TensorArray`s that represent the output.
      state: nested tuple of vector tensors that represent the state.
    Returns:
      The tuple (time + 1, output_ta_t with updated flow, new_state).
    """

    input_t = tuple(ta.read(time) for ta in input_ta)
    # Restore some shape information
    for input_, shape in zip(input_t, inputs_got_shape):
      input_.set_shape(shape[1:])

    input_t = nest.pack_sequence_as(structure=inputs, flat_sequence=input_t)
    call_cell = lambda: cell.call(input_t, state, height, width, time)

    if sequence_length is not None:
      (output, new_state) = _rnn_step(
          time=time,
          sequence_length=sequence_length,
          min_sequence_length=min_sequence_length,
          max_sequence_length=max_sequence_length,
          zero_output=zero_output,
          state=state,
          call_cell=call_cell,
          state_size=state_size,
          skip_conditionals=True)
    else:
      (output, new_state) = call_cell()

    # Pack state if using state tuples
    output = nest.flatten(output)

    output_ta_t = tuple(
        ta.write(time, out) for ta, out in zip(output_ta_t, output))

    return (time + 1, output_ta_t, new_state)

  _, output_final_ta, final_state = control_flow_ops.while_loop(
      cond=lambda time, *_: time < time_steps,
      body=_time_step,
      loop_vars=(time, output_ta, state),
      parallel_iterations=parallel_iterations,
      swap_memory=swap_memory)

  # Unpack final output if not using output tuples.
  final_outputs = tuple(ta.stack() for ta in output_final_ta)

  # Restore some shape information
  for output, output_size in zip(final_outputs, flat_output_size):
    shape = _concat(
        [const_time_steps, const_batch_size], output_size, static=True)
    output.set_shape(shape)

  final_outputs = nest.pack_sequence_as(
      structure=cell.output_size, flat_sequence=final_outputs)

  return (final_outputs, final_state)


def _rnn_step(
    time, sequence_length, min_sequence_length, max_sequence_length,
    zero_output, state, call_cell, state_size, skip_conditionals=False):
  """Calculate one step of a dynamic RNN minibatch.
  Returns an (output, state) pair conditioned on the sequence_lengths.
  When skip_conditionals=False, the pseudocode is something like:
  if t >= max_sequence_length:
    return (zero_output, state)
  if t < min_sequence_length:
    return call_cell()
  # Selectively output zeros or output, old state or new state depending
  # on if we've finished calculating each row.
  new_output, new_state = call_cell()
  final_output = np.vstack([
    zero_output if time >= sequence_lengths[r] else new_output_r
    for r, new_output_r in enumerate(new_output)
  ])
  final_state = np.vstack([
    state[r] if time >= sequence_lengths[r] else new_state_r
    for r, new_state_r in enumerate(new_state)
  ])
  return (final_output, final_state)
  Args:
    time: Python int, the current time step
    sequence_length: int32 `Tensor` vector of size [batch_size]
    min_sequence_length: int32 `Tensor` scalar, min of sequence_length
    max_sequence_length: int32 `Tensor` scalar, max of sequence_length
    zero_output: `Tensor` vector of shape [output_size]
    state: Either a single `Tensor` matrix of shape `[batch_size, state_size]`,
      or a list/tuple of such tensors.
    call_cell: lambda returning tuple of (new_output, new_state) where
      new_output is a `Tensor` matrix of shape `[batch_size, output_size]`.
      new_state is a `Tensor` matrix of shape `[batch_size, state_size]`.
    state_size: The `cell.state_size` associated with the state.
    skip_conditionals: Python bool, whether to skip using the conditional
      calculations.  This is useful for `dynamic_rnn`, where the input tensor
      matches `max_sequence_length`, and using conditionals just slows
      everything down.
  Returns:
    A tuple of (`final_output`, `final_state`) as given by the pseudocode above:
      final_output is a `Tensor` matrix of shape [batch_size, output_size]
      final_state is either a single `Tensor` matrix, or a tuple of such
        matrices (matching length and shapes of input `state`).
  Raises:
    ValueError: If the cell returns a state tuple whose length does not match
      that returned by `state_size`.
  """

  # Convert state to a list for ease of use
  flat_state = nest.flatten(state)
  flat_zero_output = nest.flatten(zero_output)

  def _copy_one_through(output, new_output):
    # If the state contains a scalar value we simply pass it through.
    if output.shape.ndims == 0:
      return new_output
    copy_cond = (time >= sequence_length)
    with ops.colocate_with(new_output):
      return array_ops.where(copy_cond, output, new_output)

  def _copy_some_through(flat_new_output, flat_new_state):
    # Use broadcasting select to determine which values should get
    # the previous state & zero output, and which values should get
    # a calculated state & output.
    flat_new_output = [
        _copy_one_through(zero_output, new_output)
        for zero_output, new_output in zip(flat_zero_output, flat_new_output)]
    flat_new_state = [
        _copy_one_through(state, new_state)
        for state, new_state in zip(flat_state, flat_new_state)]
    return flat_new_output + flat_new_state

  def _maybe_copy_some_through():
    """Run RNN step.  Pass through either no or some past state."""
    new_output, new_state = call_cell()

    nest.assert_same_structure(state, new_state)

    flat_new_state = nest.flatten(new_state)
    flat_new_output = nest.flatten(new_output)
    return control_flow_ops.cond(
        # if t < min_seq_len: calculate and return everything
        time < min_sequence_length, lambda: flat_new_output + flat_new_state,
        # else copy some of it through
        lambda: _copy_some_through(flat_new_output, flat_new_state))

  # TODO(ebrevdo): skipping these conditionals may cause a slowdown,
  # but benefits from removing cond() and its gradient.  We should
  # profile with and without this switch here.
  if skip_conditionals:
    # Instead of using conditionals, perform the selective copy at all time
    # steps.  This is faster when max_seq_len is equal to the number of unrolls
    # (which is typical for dynamic_rnn).
    new_output, new_state = call_cell()
    nest.assert_same_structure(state, new_state)
    new_state = nest.flatten(new_state)
    new_output = nest.flatten(new_output)
    final_output_and_state = _copy_some_through(new_output, new_state)
  else:
    empty_update = lambda: flat_zero_output + flat_state
    final_output_and_state = control_flow_ops.cond(
        # if t >= max_seq_len: copy all state through, output zeros
        time >= max_sequence_length, empty_update,
        # otherwise calculation is required: copy some or all of it through
        _maybe_copy_some_through)

  if len(final_output_and_state) != len(flat_zero_output) + len(flat_state):
    raise ValueError("Internal error: state and output were not concatenated "
                     "correctly.")
  final_output = final_output_and_state[:len(flat_zero_output)]
  final_state = final_output_and_state[len(flat_zero_output):]

  for output, flat_output in zip(final_output, flat_zero_output):
    output.set_shape(flat_output.get_shape())
  for substate, flat_substate in zip(final_state, flat_state):
    substate.set_shape(flat_substate.get_shape())

  final_output = nest.pack_sequence_as(
      structure=zero_output, flat_sequence=final_output)
  final_state = nest.pack_sequence_as(
      structure=state, flat_sequence=final_state)

  return final_output, final_state


def main():
    import tensorflow as tf
    import numpy as np
    tf.reset_default_graph()
    batch_size=2
    height=6
    width=5
    channels=4
    
    num_units=3
    height_extended=width+height-1
    
    inputs=tf.placeholder(tf.float32,[None, height, width, channels])
#    inputs=tf.constant(np.random.randint(0,10,[batch_size,height_extended,width,channels]), dtype=tf.float32)
#
#    inputs_t=tf.transpose(inputs,[1,0,2,3])
#    inputs_r=tf.reshape(inputs_t,[height_extended,-1,channels])
#    inputs_0=tf.reshape(inputs_r[0,:,:],[-1,channels])
    with tf.variable_scope('MDLSTM') as vs:
        cell=BasicMultidimensionalLSTMCell(num_units)
        
        state = cell.zero_state(batch_size, tf.float32)
        final_outputs=[]
    
#    for step in range(height_extended):
#        print(step)
#        outputs, state = cell.call(inputs_0, state, height, width, step)
#        final_outputs.append(outputs)
    
        outputs, state = multidimensional_dynamic_rnn(cell, inputs, height, width, dtype=tf.float32)
        weights, biases = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=vs.name)

    
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
#        print(sess.run(outputs, feed_dict={inputs: np.random.zeros(0,10,[batch_size,height,width,channels])})[0,:,:,0])
        inputs_np = np.zeros([batch_size,height,width,channels])
#        inputs_np[0,3,0,0]=1
        inputs_np[0,2,2,0]=1
        for ind in range(num_units):
            print('\n',ind,'\n')
            print(sess.run(outputs, feed_dict={inputs: inputs_np})[0,:,:,ind],'\n')
            
        print(sess.run([weights, biases], feed_dict={inputs: inputs_np}))

#        out=sess.run(final_outputs)
#        for ind in range(len(out[0])):
#            print(out[0][ind],'\n')
#        print(sess.run(inputs))
    
if __name__ == '__main__':
    main()