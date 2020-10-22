import tensorflow as tf
from abc import abstractmethod

LAYER_IDS = {}


def get_layer_id(layer_name=''):
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        return 0
    else:
        LAYER_IDS[layer_name] += 1
        return LAYER_IDS[layer_name]


class Layer(object):
    def __init__(self, name):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        self.vars = []

    def __call__(self, inputs):
        outputs = self._call(inputs)
        return outputs

    @abstractmethod
    def _call(self, inputs):
        pass


class Dense(Layer):
    def __init__(self, input_dim, output_dim, dropout=0.0, act=tf.nn.relu, name=None):
        super(Dense, self).__init__(name)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = act
        with tf.variable_scope(self.name):
            self.weight = tf.get_variable(name='weight', shape=(input_dim, output_dim), dtype=tf.float32)
            self.bias = tf.get_variable(name='bias', shape=output_dim, initializer=tf.zeros_initializer())
        self.vars = [self.weight]

    def _call(self, inputs):
        x = tf.nn.dropout(inputs, 1-self.dropout)
        output = tf.matmul(x, self.weight) + self.bias
        return self.act(output)


class CrossCompressUnit(Layer):
    def __init__(self, dim, name=None):
        super(CrossCompressUnit, self).__init__(name)
        self.dim = dim
        with tf.variable_scope(self.name):
            self.weight_v = tf.get_variable(name='weight_v', shape=(2 * dim, 1), dtype=tf.float32)
            # self.weight_e = tf.get_variable(name='weight_e', shape=(2 * dim, 1), dtype=tf.float32)
            self.bias = tf.get_variable(name='bias', shape=2 * dim, initializer=tf.zeros_initializer())
            
            self.g = tf.get_variable(name='generator_g', shape=(dim, dim), dtype=tf.float32)
            self.f = tf.get_variable(name='generator_f', shape=(dim, dim), dtype=tf.float32)
            self.gb = tf.get_variable(name='generator_g_bias', shape=dim, dtype=tf.float32)
            self.fb = tf.get_variable(name='generator_f_bias', shape=dim, dtype=tf.float32)

            self.dv = tf.get_variable(name='discriminator_v', shape=(dim, 1), dtype=tf.float32)
            self.de = tf.get_variable(name='discriminator_e', shape=(dim, 1), dtype=tf.float32)
            self.dvb = tf.get_variable(name='discriminator_v_bias', shape=1, dtype=tf.float32)
            self.deb = tf.get_variable(name='discriminator_e_bias', shape=1, dtype=tf.float32)
        self.vars = [self.weight_v, self.g, self.f, self.dv, self.de]
        # self.vars = [self.g, self.f, self.dv, self.de]

    def _generator(self, inp, direction):
        weight = self.g if direction == 've' else self.f
        bias = self.gb if direction == 've' else self.fb
        return tf.nn.relu(tf.matmul(inp, weight) + bias)

    def _discriminator(self, inp, mode):
        weight = self.dv if mode == 'v' else self.de
        bias = self.dvb if mode == 'v' else self.deb
        return tf.nn.sigmoid(tf.matmul(inp, weight) + bias)

    '''
    def _call(self, inputs):
        # [batch_size, dim]
        v, e = inputs

        fv = self._generator(e, 'ev')
        fe = self._generator(v, 've')
        
        cv = tf.concat([v, fv], 1)
        ce = tf.concat([e, fe], 1)
        xv1 = tf.expand_dims(cv, dim=2)
        xe1 = tf.expand_dims(ce, dim=1)
        xv2 = tf.expand_dims(cv, dim=1)
        xe2 = tf.expand_dims(ce, dim=2)
        m1 = tf.matmul(xv1, xe1)
        m2 = tf.matmul(xe2, xv2)
        print(">>>", cv.get_shape().as_list(), ce.get_shape().as_list(), 
            xv1.get_shape().as_list(), xe1.get_shape().as_list(), 
            xv2.get_shape().as_list(), xe2.get_shape().as_list(), 
            m1.get_shape().as_list(), m2.get_shape().as_list())
        
        outputs = tf.reshape(tf.matmul(m1, xv1) + tf.matmul(m2, xe2), [-1, 2 * self.dim])
        v_output, e_output = tf.split(outputs, 2, 1)

        return v_output, e_output, fv, fe
    '''
    '''
    def _call(self, inputs):
        # [batch_size, dim]
        v, e = inputs

        fv = self._generator(e, 'ev')
        fe = self._generator(v, 've')
        
        cv = tf.concat([v, fv], 1)
        ce = tf.concat([e, fe], 1)

        # [batch_size, dim, 1], [batch_size, 1, dim]
        cv = tf.expand_dims(cv, dim=2)
        ce = tf.expand_dims(ce, dim=1)

        # [batch_size, dim, dim]
        c_matrix = tf.matmul(cv, ce)
        c_matrix_transpose = tf.transpose(c_matrix, perm=[0, 2, 1])

        # [batch_size * dim, dim]
        c_matrix = tf.reshape(c_matrix, [-1, 2 * self.dim])
        c_matrix_transpose = tf.reshape(c_matrix_transpose, [-1, 2 * self.dim])

        # [batch_size, dim]
        outputs = tf.reshape(tf.matmul(c_matrix, self.weight_v) + tf.matmul(c_matrix_transpose, self.weight_e),
                              [-1, 2 * self.dim]) + self.bias
        v_output, e_output = tf.split(outputs, 2, 1)

        return v_output, e_output, fv, fe
'''

    def _call(self, inputs):
        # [batch_size, dim]
        v, e = inputs

        fv = self._generator(e, 'ev')
        fe = self._generator(v, 've')

        cv = tf.concat([v, fv], 1)
        ce = tf.concat([e, fe], 1)

        # [batch_size, dim, 1], [batch_size, 1, dim]
        cv = tf.expand_dims(cv, dim=2)
        ce = tf.expand_dims(ce, dim=1)

        # [batch_size, dim, dim]
        c_matrix = tf.matmul(cv, ce)
        c_matrix_transpose = tf.transpose(c_matrix, perm=[0, 2, 1])

        # [batch_size * dim, dim]
        c_matrix = tf.reshape(c_matrix, [-1, 2 * self.dim])
        c_matrix_transpose = tf.reshape(c_matrix_transpose, [-1, 2 * self.dim])

        # [batch_size, dim]
        outputs = tf.reshape(tf.matmul(c_matrix + c_matrix_transpose, self.weight_v),
                              [-1, 2 * self.dim]) + self.bias
        # outputs = tf.reshape(tf.matmul(c_matrix, self.weight_v) + tf.matmul(c_matrix_transpose, self.weight_e),
        #                       [-1, 2 * self.dim]) + self.bias
        v_output, e_output = tf.split(outputs, 2, 1)

        return v_output, e_output, fv, fe

    def get_rs_loss(self, vals):
        v, e, fv, fe = vals
        error_real = tf.reduce_mean(tf.squared_difference(self._discriminator(v, 'v'), 0.9))
        error_fake = tf.reduce_mean(tf.square(self._discriminator(fv, 'v')))
        discriminator_loss = (error_real + error_fake) / 2
        consistency_loss = tf.reduce_mean(tf.abs(self._generator(fv, 've') - e))
        rs_loss = discriminator_loss + 10 * consistency_loss
        return rs_loss

    def get_kge_loss(self, vals):
        v, e, fv, fe = vals
        error_real = tf.reduce_mean(tf.squared_difference(self._discriminator(e, 'e'), 0.9))
        error_fake = tf.reduce_mean(tf.square(self._discriminator(fe, 'e')))
        discriminator_loss = (error_real + error_fake) / 2
        consistency_loss = tf.reduce_mean(tf.abs(self._generator(fe, 'ev') - v))
        kge_loss = discriminator_loss + 10 * consistency_loss
        return kge_loss

    def get_cycle_loss(self, vals):
        v, e, fv, fe = vals
        v_real = tf.reduce_mean(tf.squared_difference(self._discriminator(v, 'v'), 0.9))
        v_fake = tf.reduce_mean(tf.square(self._discriminator(fv, 'v')))
        e_real = tf.reduce_mean(tf.squared_difference(self._discriminator(e, 'e'), 0.9))
        e_fake = tf.reduce_mean(tf.square(self._discriminator(fe, 'e')))
        discriminator_loss = (v_real + v_fake + e_real + e_fake) / 4
        consistency_loss = tf.reduce_mean(tf.abs(self._generator(fe, 'ev') - v))\
             + tf.reduce_mean(tf.abs(self._generator(fv, 've') - e))
        cycle_loss = discriminator_loss + 10 * consistency_loss
        return cycle_loss