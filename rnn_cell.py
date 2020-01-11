import tensorflow as tf


def create_single_cell(unit_type, num_units, dropout, training):
    if unit_type == "lstm":
        single_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units)
    elif unit_type == "gru":
        single_cell = tf.nn.rnn_cell.GRUCell(num_units=num_units)
    elif unit_type == "layer_norm_lstm":
        single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=num_units, layer_norm=True)
    elif unit_type == "nas":
        single_cell = tf.contrib.rnn.NASCell(num_units=num_units)
    else:
        raise ValueError("Unknown unit type %s!" % unit_type)
    if training and dropout > 0.0:
        single_cell = tf.contrib.rnn.DropoutWrapper(cell=single_cell, input_keep_prob=(1.0 - dropout))
    return single_cell
