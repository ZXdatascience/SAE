import numpy as np
import deepautoencoder.utils as utils
import tensorflow as tf

allowed_activations = ['sigmoid', 'tanh', 'softmax', 'relu', 'linear']
allowed_noises = [None, 'gaussian', 'mask']
allowed_losses = ['rmse', 'cross-entropy', 'sparse']


class StackedAutoEncoder:
    """A deep autoencoder with denoising capability"""

    def assertions(self):
        global allowed_activations, allowed_noises, allowed_losses
        assert self.loss in allowed_losses, 'Incorrect loss given'
        assert 'list' in str(
            type(self.dims)), 'dims must be a list even if there is one layer.'
        assert len(self.epoch) == len(
            self.dims), "No. of epochs must equal to no. of hidden layers"
        assert len(self.activations) == len(
            self.dims), "No. of activations must equal to no. of hidden layers"
        assert all(
            True if x > 0 else False
            for x in self.epoch), "No. of epoch must be atleast 1"
        assert set(self.activations + allowed_activations) == set(
            allowed_activations), "Incorrect activation given."
        assert utils.noise_validator(
            self.noise, allowed_noises), "Incorrect noise given"

    def __init__(self, dims, activations, epoch=1000, noise=None, loss='rmse',
                 lr=0.001, rho=0.01, beta=3, batch_size=100, print_step=50):
        self.print_step = print_step
        self.batch_size = batch_size
        self.lr = lr
        self.loss = loss
        self.activations = activations
        self.noise = noise
        self.epoch = epoch
        self.dims = dims
        self.assertions()
        self.depth = len(dims)
        self.rho = rho
        self.beta = beta
        self.weights, self.biases = [], []
        self.dense_weights, self.dense_biases = [], []
        self.dense_weights_values, self.dense_biases_values = [], []

    def add_noise(self, x):
        if self.noise == 'gaussian':
            n = np.random.normal(0, 0.1, (len(x), len(x[0])))
            return x + n
        if 'mask' in self.noise:
            frac = float(self.noise.split('-')[1])
            temp = np.copy(x)
            for i in temp:
                n = np.random.choice(len(i), round(
                    frac * len(i)), replace=False)
                i[n] = 0
            return temp
        if self.noise == 'sp':
            pass

    def fit(self, x):
        for i in range(self.depth):
            if self.noise is None:
                x = self.run(data_x=x, activation=self.activations[i],
                             data_x_=x,
                             hidden_dim=self.dims[i], epoch=self.epoch[
                                 i], loss=self.loss,
                             batch_size=self.batch_size, lr=self.lr,
                             print_step=self.print_step)
            else:
                temp = np.copy(x)
                x = self.run(data_x=self.add_noise(temp),
                             activation=self.activations[i], data_x_=x,
                             hidden_dim=self.dims[i],
                             epoch=self.epoch[
                                 i], loss=self.loss,
                             batch_size=self.batch_size,
                             lr=self.lr, print_step=self.print_step)

    def transform(self, data):
        tf.reset_default_graph()
        sess = tf.Session()
        x = tf.constant(data, dtype=tf.float32)
        for w, b, a in zip(self.weights, self.biases, self.activations):
            weight = tf.constant(w, dtype=tf.float32)
            bias = tf.constant(b, dtype=tf.float32)
            layer = tf.matmul(x, weight) + bias
            x = self.activate(layer, a)
        return x.eval(session=sess)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    @staticmethod
    def kl_divergence(rho, rho_hat):
        return rho * tf.log(rho) - rho * tf.log(rho_hat) + (1 - rho) * tf.log(1 - rho) - (1 - rho) * tf.log(1 - rho_hat)

    def run(self, data_x, data_x_, hidden_dim, activation, loss, lr,
            print_step, epoch, batch_size=100):
        tf.reset_default_graph()
        input_dim = len(data_x[0])
        sess = tf.Session()
        x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim], name='x')
        x_ = tf.placeholder(dtype=tf.float32, shape=[
                            None, input_dim], name='x_')
        encode = {'weights': tf.Variable(tf.truncated_normal(
            [input_dim, hidden_dim], dtype=tf.float32)),
            'biases': tf.Variable(tf.truncated_normal([hidden_dim],
                                                      dtype=tf.float32))}
        decode = {'biases': tf.Variable(tf.truncated_normal([input_dim],
                                                            dtype=tf.float32)),
                  'weights': tf.transpose(encode['weights'])}
        encoded = self.activate(
            tf.matmul(x, encode['weights']) + encode['biases'], activation)
        decoded = self.activate(tf.matmul(encoded, decode['weights']) + decode['biases'], activation)

        # reconstruction loss
        if loss == 'rmse':
            loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(x_, decoded))))
        elif loss == 'cross-entropy':
            loss = -tf.reduce_mean(x_ * tf.log(decoded))
        elif loss == 'sparse':
            rho_hat = tf.reduce_mean(encoded, axis=0)
            kl = self.kl_divergence(self.rho, rho_hat)
            diff = tf.subtract(x_, decoded)
            loss = 0.5 * tf.reduce_mean(tf.reduce_sum(diff ** 2, axis=1)) + self.beta * tf.reduce_sum(kl)
        train_op = tf.train.AdamOptimizer(lr).minimize(loss)

        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            b_x, b_x_ = utils.get_batch(
                data_x, data_x_, batch_size)
            sess.run(train_op, feed_dict={x: b_x, x_: b_x_})
            if (i + 1) % print_step == 0:
                l = sess.run(loss, feed_dict={x: data_x, x_: data_x_})
                # print('epoch {0}: global loss = {1}'.format(i, l))
        # self.loss_val = l
        # debug
        # print('Decoded', sess.run(decoded, feed_dict={x: self.data_x_})[0])
        self.weights.append(sess.run(encode['weights']))
        self.biases.append(sess.run(encode['biases']))
        return sess.run(encoded, feed_dict={x: data_x_})


    def finetunning(self, data, labels, loss, dense_activations, dense_layers, learning_rate, print_step, epoch,
                    ae_traininable=True, batch_size=100):
        tf.reset_default_graph()
        sess = tf.Session()
        input_dim = len(data[0])
        x = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_dim], name='x')
        new_x = x
        target = tf.placeholder(dtype=tf.float32, shape=[batch_size, 1], name='target')
        for w, b, a in zip(self.weights, self.biases, self.activations):
            weight = tf.Variable(initial_value=w, dtype=tf.float32, name='ae_weight', trainable=ae_traininable)
            bias = tf.Variable(initial_value=b, dtype=tf.float32, name='ae_bias', trainable=ae_traininable)
            new_x = self.activate(tf.add(tf.matmul(new_x, weight), bias), a)

        for i, layer in enumerate(dense_layers):
            if i == 0:
                input_dim = self.dims[-1]
            else:
                input_dim = dense_layers[i-1]
            weight_dense = tf.Variable(tf.truncated_normal([input_dim, layer]), dtype=tf.float32, name='dense_weight')
            bias_dense = tf.Variable(tf.truncated_normal([layer]), dtype=tf.float32, name='dense_bias')
            new_x = self.activate(tf.add(tf.matmul(new_x, weight_dense), bias_dense), dense_activations[i])
            self.dense_weights.append(weight_dense)
            self.dense_biases.append(bias_dense)
        if loss == 'rmse':
            loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(new_x, target))))
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        print(tf.trainable_variables())
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            b_x, b_x_ = utils.get_batch(
                data, labels, batch_size)
            b_x_ = b_x_.reshape(-1, 1)
            sess.run(train_op, feed_dict={x: b_x, target: b_x_})
            if (i + 1) % print_step == 0:
                l, o, t = sess.run([loss, new_x, target], feed_dict={x: b_x, target: b_x_})
                print('epoch {0}: loss = {1}'.format(i, l))
                # print('epoch {0}: loss = {1}, output={2}, target={3}'.format(i, l, o.reshape(-1), t.reshape(-1)))
        for i in range(len(dense_layers)):
            self.dense_weights_values.append(sess.run(self.dense_weights[i]))
            self.dense_biases_values.append(sess.run(self.dense_biases[i]))

    def dense_evaluate(self, data, activations):
        tf.reset_default_graph()
        sess = tf.Session()
        x = tf.constant(data, dtype=tf.float32)
        for w_ae, b_ae, a in zip(self.weights, self.biases, self.activations):
            weight_ae = tf.constant(w_ae, dtype=tf.float32)
            bias_ae = tf.constant(b_ae, dtype=tf.float32)
            layer = tf.matmul(x, weight_ae) + bias_ae
            x = self.activate(layer, a)
        for w_dense, b_dense, a_dense in zip(self.dense_weights_values, self.dense_biases_values, activations):
            weight_dense = tf.constant(w_dense)
            bias_dense = tf.constant(b_dense)
            x = self.activate(tf.matmul(x, weight_dense) + bias_dense, a_dense)
            print(x)
        test_res = x.eval(session=sess)
        return test_res


    def activate(self, linear, name):
        if name == 'sigmoid':
            return tf.nn.sigmoid(linear, name='encoded')
        elif name == 'softmax':
            return tf.nn.softmax(linear, name='encoded')
        elif name == 'linear':
            return linear
        elif name == 'tanh':
            return tf.nn.tanh(linear, name='encoded')
        elif name == 'relu':
            return tf.nn.relu(linear, name='encoded')
