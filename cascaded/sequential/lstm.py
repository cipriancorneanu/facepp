from sequential import Sequential
import numpy as np
import tensorflow as tf
import random


class SequentialLstm(Sequential):
    def __init__(self, aligner, encoder=None, window_size=100, num_cells=100, num_hidden=75, recurrence='lstm', learning_rate=0.01, gamma=0.9, momentum=0.0):
        Sequential.__init__(self, aligner, encoder=encoder)
        self.window_size = window_size
        self.num_cells, self.num_hidden = num_cells, num_hidden
        self.recurrence = recurrence
        self.states = None
        self.hidden = None
        self.old_geometries = None
        self.out_scales = None
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.momentum = momentum

        # Build model variables, prepare tensorflow session & initialize variables
        self.vars = self._construct_variables()
        self.fcns = self._construct_functions()
        self.session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.session.run(tf.initialize_all_variables())

    def __getstate__(self):
        """ On pickling avoid saving the session """
        state = self.__dict__.copy()
        state['vars'] = None
        state['fcns'] = None
        state['session'] = None
        return state

    def __setstate__(self, d):
        """ On unpickling, create a new session and start """
        self.__dict__ = d
        self.vars = self._construct_variables()
        self.fcns = self._construct_functions()
        self.session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.session.run(tf.initialize_all_variables())

    def save(self, file):
        Sequential.save(self, file)
        tf.train.Saver().save(self.session, file + '_model.ckpt')

    @staticmethod
    def load(file):
        model = Sequential.load(file)
        tf.train.Saver().restore(model.session, file + '_model.ckpt')
        return model

    # Parent class access points
    # --------------------------------------------------

    def _train(self, sequences, geometries, save_file=None):
        deltas = [self.encoder.encode_deltas(g[:-1, ...], g[1:, ...]) for g in geometries]  # Calculate delta parameters
        self.out_means = np.mean(np.concatenate(deltas, axis=0), axis=0)  # Find delta parameters mean
        self.out_scales = np.std(np.concatenate(deltas, axis=0), axis=0)  # Find delta parameters stdv.
        deltas = [(g - self.out_means) / self.out_scales[None, :] for g in deltas]  # Standardize delta parameters
        deltas = [np.pad(d, ((1, 0), (0, 0)), 'constant', constant_values=(0,)) for d in deltas]  # Pad deltas

        # Perform training
        self._train_lstm(deltas, save_file=save_file)

    def _initialize_memory(self):
        # Prepare hidden states & outputs
        self.states = np.zeros((len(self.geometries), self.num_cells), dtype=np.float32)
        self.hidden = np.zeros((len(self.geometries), self.num_cells), dtype=np.float32)

    def _predict_geometry(self, frames, indices):
        # Prepare inputs
        inputs = np.zeros(
            (len(indices), self.encoder.get_parameters_length()), dtype=np.float32
        ) if self.old_geometries is None else ((self.encoder.encode_deltas(
            self.old_geometries[indices, ...], self.geometries[indices, ...]
        ) - self.out_means) / self.out_scales[None, :])

        # Update old geometries & make predictions
        self.old_geometries = np.copy(self.geometries)
        outputs = self._test_lstm(indices, inputs) * self.out_scales[None, :] + self.out_means
        return self.encoder.decode_deltas(self.encoder.encode_parameters(self.geometries[indices, ...]), outputs)

    # LSTM construction methods
    # --------------------------------------------------

    def _construct_variables(self):
        factor = {'lstm': 4, 'gru': 3}[self.recurrence]
        n_outputs = self.encoder.get_parameters_length()
        weights_lstm = SequentialLstm._weights_orthoglorot((self.num_cells + n_outputs, factor * self.num_cells))
        weights_fc = SequentialLstm._weights_orthoglorot((self.num_cells, n_outputs))
        return {
            # Define lstm layer parameters
            'wl_h': tf.Variable(
                weights_lstm[:self.num_cells, ...],
                dtype=tf.float32,
                name='w',
            ), 'wl_x': tf.Variable(
                weights_lstm[self.num_cells:, ...],
                dtype=tf.float32,
                name='w',
            ), 'bl': tf.Variable(
                np.zeros((factor*self.num_cells,), dtype=np.float32),
                name='b',
            ),

            # Define fc layer parameters
            'wf': tf.Variable(
                weights_fc,
                dtype=tf.float32,
                name='w',
            ), 'bf': tf.Variable(
                np.zeros((n_outputs,), dtype=np.float32),
                name='b',
            ),
        }

    def _construct_functions(self):
        return {
            'train': self._construct_function_train(),
            'test': self._construct_function_test()
        }

    def _construct_function_train(self):
        # Prepare placeholders & outputs
        # ----------------------------------------------------

        placeholders = [tf.placeholder(
            tf.float32, shape=(None, self.encoder.get_parameters_length())
        ) for _ in range(self.window_size)]
        results = [None] * (len(placeholders) - 1)

        # Define pipeline
        recurrence = getattr(self, '_build_' + self.recurrence)
        out, hid = recurrence(None, None, placeholders[0])
        results[0] = self._build_fc(out)
        for i in range(1, self.window_size-1):
            out, hid = recurrence(hid, out, placeholders[i])
            results[i] = self._build_fc(out)

        # Define loss function & error
        loss_fcn = tf.add_n([tf.reduce_mean(tf.abs(r - g)) for r, g in zip(results, placeholders[1:])]) / (
            self.window_size - 1
        )

        # Prepare placeholder variables
        learning_rate = tf.placeholder(tf.float32)
        momentum = tf.placeholder(tf.float32)
        gamma = tf.placeholder(tf.float32)

        # Prepare optimizer (SGD)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # optimize_function = optimizer.minimize(loss_fcn)

        # Prepare optimizer (RmsProp)
        optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=momentum, decay=gamma)
        optimize_function = optimizer.minimize(loss_fcn)

        def _train_fcn(data):
            feed_dict = {p: v for p, v in zip(placeholders, data)}
            feed_dict[learning_rate] = self.learning_rate
            feed_dict[momentum] = self.momentum
            feed_dict[gamma] = self.gamma
            ret = self.session.run([optimize_function, loss_fcn], feed_dict=feed_dict)
            return ret[1]

        return _train_fcn

    def _construct_function_test(self):
        # Prepare placeholders
        plh_state = tf.placeholder(tf.float32, shape=(None, self.num_cells))
        plh_hiden = tf.placeholder(tf.float32, shape=(None, self.num_cells))
        plh_input = tf.placeholder(tf.float32, shape=(None, self.encoder.get_parameters_length()))

        # Define gate outputs
        recurrence = getattr(self, '_build_'+self.recurrence)
        _h, _state = recurrence(plh_state, plh_hiden, plh_input)
        _out = self._build_fc(_h)

        def _test_fcn(state, h, x):
            return self.session.run([_state, _h, _out], feed_dict={plh_state: state, plh_hiden: h, plh_input: x})

        return _test_fcn

    # GRU layer definition
    def _build_gru(self, state, h, x):
        # Calculate outputs of gates 1 & 2
        g12_h = tf.matmul(h, self.vars['wl_h'][:, :2*self.num_cells]) if h is not None else 0
        g12_x = tf.matmul(x, self.vars['wl_x'][:, :2*self.num_cells])
        g12 = tf.nn.sigmoid(tf.nn.bias_add(g12_h+g12_x, self.vars['bl'][:2*self.num_cells]))

        # Calculate output of gate 3
        h2 = (g12[:, :self.num_cells] * h) if h is not None else 0
        g3_h = tf.matmul(h2, self.vars['wl_h'][:, 2*self.num_cells:3*self.num_cells]) if h is not None else 0
        g3_x = tf.matmul(x, self.vars['wl_x'][:, 2*self.num_cells:3*self.num_cells])
        g3 = tf.nn.tanh(tf.nn.bias_add(g3_h+g3_x, self.vars['bl'][2*self.num_cells:3*self.num_cells]))

        # Calculate updated hidden state
        h = g3 * g12[:, self.num_cells:] + (0 if h is None else h) * (1 - g12[:, self.num_cells:])
        return h, h

    # LSTM layer definition
    def _build_lstm(self, state, h, x):
        # Define gate outputs
        h_prod, x_prod = (tf.matmul(h, self.vars['wl_h']) if h is not None else 0), tf.matmul(x, self.vars['wl_x'])
        gates = tf.split(1, 4, tf.nn.sigmoid(tf.nn.bias_add(h_prod + x_prod, self.vars['bl'])))

        # Update internal state and output (h)
        state = ((state * gates[0]) if state is not None else 0) + (gates[1] * (2 * gates[2] - 1))
        h = tf.nn.tanh(state) * gates[3]
        return h, state

    # FC layer definition
    def _build_fc(self, input):
        return tf.nn.bias_add(tf.matmul(input, self.vars['wf']), self.vars['bf'])

    @staticmethod
    def _weights_orthoglorot(shape, factor=1):
        """ ortho: Initialize weights with orthogonal initialization (consider inputs variance) """
        flat_shape = (np.prod(shape[:-1]), shape[-1])
        u, _, v = np.linalg.svd(np.random.randn(*shape), full_matrices=False)

        # Pick the one with the correct shape
        q = u if u.shape == flat_shape else v
        return (factor * q).reshape(shape).astype(np.float32)

    # LSTM execution methods
    # --------------------------------------------------

    def _train_lstm(self, train, save_file=None):
        for i in range(1, 10000+1):
            loss = self.fcns['train'](self._sample_data(train))
            if i % 1000 == 0 and save_file is not None:
                self.save(save_file)
            print 'Iteration ' + str(i) + ' loss=' + str(loss)

    def _test_lstm(self, indices, inputs):
        states, hidden, outputs = self.fcns['test'](
            self.states[indices, ...],
            self.hidden[indices, ...],
            inputs
        )

        self.states[indices, ...] = states
        self.hidden[indices, ...] = hidden
        return outputs

    def _sample_data(self, x):
        # x = [x[i] for i in random.sample(xrange(len(x)), min(128, len(x)))]
        inds = [random.randrange(len(e)-self.window_size) for e in x]
        ret = np.array([e[i:i+self.window_size, ...] for i, e in zip(inds, x)])
        return np.swapaxes(ret, 0, 1)
