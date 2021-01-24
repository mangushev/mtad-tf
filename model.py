
#TODO:
#check residual connections
#dropout: gat
#normalization
#attention calculation

import tensorflow as tf

import sys
import math
import six

import numpy as np

def dropout(input_tensor, dropout_prob):
  """Perform dropout.

  Args:
    input_tensor: float Tensor.
    dropout_prob: Python float. The probability of dropping out a value (NOT of
      *keeping* a dimension as in `tf.nn.dropout`).

  Returns:
    A version of `input_tensor` with dropout applied.
  """
  if dropout_prob is None or dropout_prob == 0.0:
    return input_tensor

  output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
  return output


def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if name is None:
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape

def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.get_variable_scope().name
    raise ValueError(
        "For the tensor `%s` in scope `%s`, the actual rank "
        "`%d` (shape = %s) is not equal to the expected rank `%s`" %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))

def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.compat.v1.truncated_normal_initializer(stddev=initializer_range)

#inputs: batch_shape + [in_width, in_channels]
#filter: [filter_width, in_channels, out_channels]
def conv1d_layer(inputs, filter_width, in_channels, out_channels, padding, activation, initializer, trainable=True, name="conv"):
  with tf.compat.v1.variable_scope(name):
    filter = tf.compat.v1.get_variable(initializer=initializer, shape=[filter_width, in_channels, out_channels], trainable=trainable, name='filter')
    conv = tf.nn.conv1d(inputs, filter, [1], padding=padding, name="conv")
    bias = tf.compat.v1.get_variable(initializer=tf.zeros_initializer, shape=[out_channels], trainable=trainable, name='bias')
    conv_bias = tf.nn.bias_add(conv, bias, name='conv_bias')
    conv_bias_relu = activation(conv_bias, name='conv_bias_relu')
    return conv_bias_relu

def dense_layer(input_tensor, hidden_size, activation, initializer, name="dense"):
  with tf.compat.v1.variable_scope(name):
    input_shape = get_shape_list(input_tensor, expected_rank=2)
    batch_size = input_shape[0]
    input_width = input_shape[1]

    w = tf.compat.v1.get_variable(initializer=initializer, shape=[input_width, hidden_size], name="w")
    z = tf.matmul(input_tensor, w)
    b = tf.compat.v1.get_variable(initializer=tf.zeros_initializer, shape=[hidden_size], name="b")
    y = tf.nn.bias_add(z, b)
    if (activation):
      y = activation(y)
    return y

def layer_norm(input_tensor, trainable=True, name=None):
  """Run layer normalization on the last dimension of the tensor."""
  #return tf.keras.layers.LayerNormalization(name=name,trainable=trainable,axis=-1,epsilon=1e-14,dtype=tf.float32)(input_tensor)
  return tf.contrib.layers.layer_norm(
      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, trainable=trainable, scope=name)

def print_shape(tensor, rank, tensor_name):
  return tensor
  tensor_shape = get_shape_list(tensor, expected_rank=rank)
  return tf.Print(tensor, [tensor_shape], tensor_name, summarize=8)

def Block(input_tensor, num_features, window_size, activation_fn, initializer_range, block_id, dropout_prob=0.1):

  with tf.variable_scope("tc_module_%d" %block_id):
    tc_output_1 = conv1d_layer(input_tensor, 3, num_features, num_features, 'SAME', activation_fn, 
       create_initializer(initializer_range), name="conv_1")
    tc_output_1 = dropout(tc_output_1, dropout_prob)
    tc_output_2 = conv1d_layer(input_tensor, 5, num_features, num_features, 'SAME', activation_fn, 
       create_initializer(initializer_range), name="conv_2")
    tc_output_2 = dropout(tc_output_2, dropout_prob)
    tc_output_3 = conv1d_layer(input_tensor, 7, num_features, num_features, 'SAME', activation_fn, 
       create_initializer(initializer_range), name="conv_3")
    tc_output_3 = dropout(tc_output_3, dropout_prob)
    tc_output_4 = conv1d_layer(input_tensor, 9, num_features, num_features, 'SAME', activation_fn, 
       create_initializer(initializer_range), name="conv_4")
    tc_output_4 = dropout(tc_output_4, dropout_prob)

    #[A, w, m] --> [A, w, m]
    tc_output1 = (tc_output_1 + tc_output_2 + tc_output_3 + tc_output_4) / 4
    tc_output = print_shape(tc_output1, 3, "tc_output shape")

  with tf.variable_scope("gat_module_%d" %block_id):

    #[A, w, m] --> [A, m, w]
    gat_input = tf.transpose(tc_output, [0, 2, 1])

    #[m, w]
    gat_weights = tf.compat.v1.get_variable(initializer=create_initializer(initializer_range), 
          shape=[2 * window_size, 1], name='gat_weights')

    #[A, m, w] --> [A, m, 1, w]
    i_dim1 = tf.reshape(gat_input, [-1, num_features, 1, window_size], name='i_dim1')
    #[A, m, 1, w] --> [A, m, m, w]
    i_dim = tf.tile(i_dim1, [1, 1, num_features, 1])
    j_dim = tf.transpose(i_dim, [0, 2, 1, 3])

    #[A, m, m, w] + [A, m, m, w] --> [A, m, m, 2w]
    ij_concat_p = tf.concat([i_dim, j_dim], axis=-1) 
    ij_concat = print_shape(ij_concat_p, 4, "ij_concat shape")

    w1 = tf.reshape(gat_weights, [1, 1, 2 * window_size, 1], name='w1')
    #[A, 1, 1, 2w, 1] --> [A, m, m, 2w, 1]
    ij_concat_1 = tf.reshape(ij_concat, [-1, num_features, num_features, 2 * window_size, 1], name='ij_concat_1')

    #[2w, 1]' . [A, m, m, 2w, 1] --> [A, m, m, 1, 1]
    mm_p = tf.matmul(gat_weights, ij_concat_1, transpose_a=True)
    mm = print_shape(mm_p, 5, "mm shape")
 
    #[A, m, m, 1, 1] --> [A, m, m]
    mm = tf.squeeze(mm, axis=-1, name='squeeze1')
    mm = tf.squeeze(mm, axis=-1, name='squeeze2')

    e_ij = tf.nn.leaky_relu(mm, alpha=0.2, name='e_ij')

    #[A, m, m] --> [A, m, m]
    alpha_ij_p = tf.nn.softmax(e_ij, axis=1)
    alpha_ij = print_shape(alpha_ij_p, 3, "alpha_ij shape")

    #[A, m, w] --> [A, m, 1, w]
    x1 = tf.reshape(gat_input, [-1, num_features, 1, window_size])
    #[A, m, 1, w] --> [A, 1, m, w]
    x2 = tf.transpose(x1, [0, 2, 1, 3])
    #[A, 1, m, w] --> [A, m, m, w]
    x3 = tf.tile(x2, [1, num_features, 1, 1])
    #[A, m, m] --> [A, m, m, 1]
    alpha_ij_1 = tf.reshape(alpha_ij, [-1, num_features, num_features, 1])
    #[A, m, m, 1] * [A, m, m, w] --> reduce_sum axis=2 --> [A, m, w]
    h_i_p = tf.math.sigmoid(tf.reduce_sum(alpha_ij_1 * x3, axis=2))
    h_i = print_shape(h_i_p, 3, "h_i shape")

    #[A, m, w] --> [A, w, m]
    gat_output = tf.transpose(h_i, [0, 2, 1])

    return gat_output 

class MtadTF(object):
  #   A - batch size
  #   m = number of variables or features (metrics for computer instance)
  #   w = window size
  #   k0 = conv1d filter width 
  #   k1 = hidden dimension of the GRU layer
  #   k2 = hidden dimension of fully connected layers
  def __init__(self,
               input_tensor,
               conv1d_act_fn=tf.nn.relu,
               k0=7,
               k1=150,
               k2=150,
               tc_act_fn=tf.nn.relu,
               gru_act_fn=tf.nn.relu,
               initializer_range=0.02,
               dropout_prob=0.1,
               is_training=True):
    #[A, w, m]
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    window_size = input_shape[1]
    num_features = input_shape[2]

    if is_training == False:
       dropout_prob = 0.0

    input_t = print_shape(input_tensor, 3, "input_tensor shape")

    #1). 1D convolution to alleviate the possible noise effects
    #[A, w, m] --> [A, w, m]
    with tf.variable_scope("alleviate_noise"):
      conv1d_output1 = conv1d_layer(input_t, k0, num_features, num_features, "SAME", 
          conv1d_act_fn, create_initializer(initializer_range), name="conv_1")
      conv1d_output_p = dropout(conv1d_output1, dropout_prob)
      conv1d_output = print_shape(conv1d_output_p, 3, "conv1d_output shape")

    #2). temporal convolution and grath attention
    #[A, w, m] --> [A, w, m]
    with tf.variable_scope("blocks"):
      block1_output = Block(conv1d_output, num_features, window_size, tc_act_fn, initializer_range, 1, dropout_prob)
      block2_output = Block(conv1d_output + block1_output, num_features, window_size, tc_act_fn, initializer_range, 2, dropout_prob)
      block3_output = Block(conv1d_output + block2_output, num_features, window_size, tc_act_fn, initializer_range, 3, dropout_prob)

      #[A, w, m] concat [A, w, m] concat [A, w, m] --> [A, w, 3m]
      all_block_output1 = tf.concat([block1_output, block2_output, block3_output, conv1d_output], 2, name='concat')
      all_block_output = print_shape(all_block_output1, 3, "all_block_output shape")

    #3). GRU to model time series in addition to TC (temporal convolutions model)
    #[A, w, 4m] --> [A, w, k1]
    with tf.name_scope('GRU') as scope:
      #[A, w, 4m] -> [w, A, 4m]
      step_inputs = tf.transpose(all_block_output, [1, 0, 2])

      with tf.compat.v1.variable_scope('gru_cells'):

        gru_cell = tf.keras.layers.GRUCell(k1, activation=gru_act_fn, kernel_initializer=tf.compat.v1.initializers.he_normal(), recurrent_initializer=tf.orthogonal_initializer, bias_initializer=tf.zeros_initializer, dropout=dropout_prob, name='gru_cell')

        step = tf.constant(0)
        output_ta = tf.TensorArray(size=window_size, dtype=tf.float32)
        initial_state = tf.zeros((batch_size, k1), dtype=tf.float32, name='state')

        def cond(step, output_ta, state):
          return tf.less(step, window_size)

        def body(step, output_ta, state):
          input = tf.slice(step_inputs, [step, 0, 0], [1, -1, -1], name='slice')
          input_one = tf.squeeze(input, axis=0, name='squeeze')
          output, state = gru_cell(input_one, state, training=is_training)

          output_ta = output_ta.write(step, output, name='ta_w')

          return (step + 1, output_ta, state)

        _, output_ta_final, state = tf.while_loop(cond, body, [step, output_ta, [initial_state]], name='gru_loop')

      #time, batch, features: add outputs as per article
      time_gru_output = output_ta_final.stack(name='stack_ta')
   
      #[w, A, k1] -> [A, w, k1]
      gru_output = tf.transpose(time_gru_output, [1, 0, 2])

    #4). three fully connected layers
    #[A, w, k1] --> [A, k1]
    final_gru_output = gru_output[:, -1, :]
    #final_gru_output = tf.squeeze(final_gru_output1) 

    #[A, k1] --> [A, k2]
    with tf.variable_scope("layer_3fc"):
      layer_output = final_gru_output
      for i in range(3):
        with tf.variable_scope("fc_%d" %i):
          layer_output = dense_layer(layer_output, k2, activation=None, initializer=create_initializer(initializer_range))
          #layer_output = layer_norm(layer_output)
          layer_output = tf.nn.relu(layer_output)
          layer_output = dropout(layer_output, dropout_prob)
  
    #[A, k2] --> [A, m]
    self._next_feature = dense_layer(layer_output, num_features, activation=None, initializer=create_initializer(initializer_range))


  @property
  def next_feature(self):
    return self._next_feature
