import tensorflow as tf


class StackedRnn:
    """학습 모델을 정의한다."""

    def __init__(self, params):
        self.params = params

    def get_stacted_rnn_model(self):
        """Stacted RNN Model을 그린다."""
        seq_length = self.params['seq_length']
        data_dim = self.params['data_dim']
        hidden_dims = self.params['hidden_dims']

        tf.reset_default_graph()
        X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
        X_closes = tf.placeholder(tf.float32, [None, 1])
        Y = tf.placeholder(tf.float32, [None, 1])
        output_keep_prob = tf.placeholder(tf.float32)

        cells = []
        for n in hidden_dims:
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=n, activation=tf.tanh)
            dropout_cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=output_keep_prob)
            cells.append(dropout_cell)
        stacked_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        outputs, _states = tf.nn.dynamic_rnn(stacked_rnn_cell, X, dtype=tf.float32)
        Y_pred = tf.contrib.layers.fully_connected(
            outputs[:, -1], self.params['output_dim'], activation_fn=None)  # We use the last cell's output

        # cost/loss
        loss = tf.reduce_sum(tf.square((Y - Y_pred) / (1 + Y - X_closes)))

        optimizer = tf.train.AdamOptimizer(self.params['learning_rate'])
        train = optimizer.minimize(loss)

        # RMSE
        targets = tf.placeholder(tf.float32, [None, 1])
        predictions = tf.placeholder(tf.float32, [None, 1])
        rmse = tf.sqrt(tf.reduce_mean(tf.square((targets - predictions) / (1 + targets - X_closes))))

        return {
            'X': X,
            'Y': Y,
            'output_keep_prob': output_keep_prob,
            'train': train,
            'loss': loss,
            'Y_pred': Y_pred,
            'targets': targets,
            'rmse': rmse,
            'predictions': predictions,
            'X_closes': X_closes
        }