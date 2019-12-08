import tensorflow as tf

keras = tf.keras


class ListNet(keras.Model):
    def __init__(self):
        super(ListNet, self).__init__(self)
        l2_reg = 0.01

        self.dense1 = keras.layers.Dense(
            32, activation="relu", kernel_regularizer=keras.regularizers.l2(l2_reg)
        )
        self.dense2 = keras.layers.Dense(
            32, activation="relu", kernel_regularizer=keras.regularizers.l2(l2_reg)
        )
        self.dense_output = keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)

        return self.dense_output(x)
