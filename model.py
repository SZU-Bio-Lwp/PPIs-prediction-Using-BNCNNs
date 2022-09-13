import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()


def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


class CNNs_Model(object):
    def __init__(self, height=15, width=20, num_label=2, batch_size=64):
        """
        Args:
            height: The height of inputs.
            width: The width of inputs.
            num_label: The category number.
            batch_size: The size of one training batch
        """

        self.height = height
        self.width = width
        self.Input_dim = height * width
        self.Output_dim = num_label
        self.batch_size = batch_size

        # model definition
        self.x = tf.compat.v1.placeholder(tf.float32, [None, self.Input_dim])
        self.y = tf.compat.v1.placeholder(tf.float32, [None, self.Output_dim])

        self.x_input = tf.reshape(self.x, [-1, height, width, 1])
        self.W_conv1 = weight_variable([3, 3, 1, 32])
        self.b_conv1 = bias_variable([32])

        self.h_conv1 = tf.nn.softplus(conv2d(self.x_input, self.W_conv1) + self.b_conv1)
        self.bn1 = tf.keras.layers.BatchNormalization()(self.h_conv1, training=True)
        self.h_pool1 = max_pool_2x2(self.bn1)

        self.W_conv2 = weight_variable([3, 3, 32, 64])
        self.b_conv2 = bias_variable([64])

        self.h_conv2 = tf.nn.softplus(conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
        self.bn2 = tf.keras.layers.BatchNormalization()(self.h_conv2, training=True)
        self.h_pool2 = max_pool_2x2(self.bn2)

        self.W_fc1 = weight_variable([1 * 2 * 64, 1024])
        self.b_fc1 = bias_variable([1024])

        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 1 * 2 * 64])
        self.h_fc1 = tf.nn.softplus(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)

        self.drop_prob = tf.compat.v1.placeholder("float")
        self.h_fc1_dropout = tf.nn.dropout(self.h_fc1, rate=self.drop_prob)

        # output
        self.W_fc2 = weight_variable([1024, self.Output_dim])
        self.b_fc2 = bias_variable([self.Output_dim])
        self.y_conv = tf.matmul(self.h_fc1_dropout, self.W_fc2) + self.b_fc2
        self.pred = tf.nn.softmax(self.y_conv)

        self.learning_rate = 0.01
        self.global_step = tf.Variable(0, trainable=False)
        self.decaylearning_rate = tf.compat.v1.train.exponential_decay(self.learning_rate, self.global_step, 10, 0.9)
        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.y_conv))
        self.train_step = tf.compat.v1.train.AdamOptimizer(self.decaylearning_rate).minimize(self.cross_entropy)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_conv, 1)), tf.float32))

