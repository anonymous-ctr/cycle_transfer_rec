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
            #self.weight_ve = tf.get_variable(name='weight_ve', shape=(1), dtype=tf.float32)
            #self.weight_vv = tf.get_variable(name='weight_vv', shape=(1), dtype=tf.float32)
            #self.weight_ee = tf.get_variable(name='weight_ee', shape=(1), dtype=tf.float32)
            #self.weight_ev = tf.get_variable(name='weight_ev', shape=(1), dtype=tf.float32)
            #self.weight_v = tf.get_variable(name='weight_v', shape=(dim), dtype=tf.float32)
            #self.weight_e = tf.get_variable(name='weight_e', shape=(dim), dtype=tf.float32)
            self.weight_mvv = tf.get_variable(name='weight_mvv', shape=(dim), dtype=tf.float32)
            self.weight_mve = tf.get_variable(name='weight_mve', shape=(dim), dtype=tf.float32)
            self.weight_mee = tf.get_variable(name='weight_mee', shape=(dim), dtype=tf.float32)
            self.weight_mev = tf.get_variable(name='weight_mev', shape=(dim), dtype=tf.float32)
            self.bias_v = tf.get_variable(name='bias_v', shape=dim, initializer=tf.zeros_initializer())
            self.bias_e = tf.get_variable(name='bias_e', shape=dim, initializer=tf.zeros_initializer())
            
            self.g = tf.get_variable(name='generator_g', shape=(dim, dim), dtype=tf.float32)
            self.f = tf.get_variable(name='generator_f', shape=(dim, dim), dtype=tf.float32)
            self.gb = tf.get_variable(name='generator_g_bias', shape=dim, dtype=tf.float32)
            self.fb = tf.get_variable(name='generator_f_bias', shape=dim, dtype=tf.float32)

            self.dv = tf.get_variable(name='discriminator_v', shape=(dim, 1), dtype=tf.float32)
            self.de = tf.get_variable(name='discriminator_e', shape=(dim, 1), dtype=tf.float32)
            self.dvb = tf.get_variable(name='discriminator_v_bias', shape=1, dtype=tf.float32)
            self.deb = tf.get_variable(name='discriminator_e_bias', shape=1, dtype=tf.float32)
        #self.vars = [self.weight_v, self.weight_e, self.g, self.f, self.dv, self.de]
        self.vars = [
                #self.weight_v,
                #self.weight_e,
                #self.weight_vv, 
                #self.weight_ve, 
                #self.weight_ev, 
                #self.weight_ee, 
                self.weight_mvv, 
                self.weight_mve, 
                self.weight_mev, 
                self.weight_mee, 
                self.g, self.f, self.dv, self.de]

    def _generator(self, inp, direction):
        weight = self.g if direction == 've' else self.f
        bias = self.gb if direction == 've' else self.fb
        return tf.nn.relu(tf.matmul(inp, weight) + bias)

    def _discriminator(self, inp, mode):
        weight = self.dv if mode == 'v' else self.de
        bias = self.dvb if mode == 'v' else self.deb
        return tf.nn.sigmoid(tf.matmul(inp, weight) + bias)

    def _call(self, inputs):
        # [batch_size, dim]
        v, e = inputs

        fv = self._generator(e, 'ev')
        fe = self._generator(v, 've')

        #v_matrix = tf.matmul(tf.expand_dims(v, dim=2), tf.expand_dims(fv, dim=1))
        #vt_matrix = tf.transpose(v_matrix, perm=[0, 2, 1])
        #v_output = tf.reshape(tf.matmul(v_matrix, self.weight_vv) + tf.matmul(vt_matrix, self.weight_ve), [-1, self.dim]) + self.bias_v
        #v_output = tf.multiply(v, self.weight_mvv) + tf.multiply(fv, self.weight_mve) + self.bias_v
                #+ tf.multiply(tf.multiply(v, fv), self.weight_v)
                #+ tf.reshape(tf.matmul(v_matrix, self.weight_vv)\
                #+ tf.matmul(vt_matrix, self.weight_ve)\
                #, [-1, self.dim])
        #v_output = tf.multiply(self.weight_mvv, v + fv) + self.bias_v
        v_output = tf.multiply(self.weight_mvv, v) + tf.multiply(self.weight_mve, fv) + self.bias_v

        #e_matrix = tf.matmul(tf.expand_dims(e, dim=2), tf.expand_dims(fe, dim=1))
        #et_matrix = tf.transpose(e_matrix, perm=[0, 2, 1])
        #e_output = tf.reshape(tf.matmul(e_matrix, self.weight_ee) + tf.matmul(et_matrix, self.weight_ev), [-1, self.dim]) + self.bias_e
        #e_output = tf.multiply(e, self.weight_mee) + tf.multiply(fe, self.weight_mev) + self.bias_e
                #+ tf.multiply(tf.multiply(e, fe), self.weight_e)
                #+ tf.reshape(tf.matmul(e_matrix, self.weight_ee)\
                #+ tf.matmul(et_matrix, self.weight_ev) \
                #, [-1, self.dim])
        #e_output = tf.multiply(self.weight_mee, e + fe) + self.bias_e
        e_output = tf.multiply(self.weight_mee, e) + tf.multiply(self.weight_mev, fe) + self.bias_e

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
