import tensorflow as tf

class Discriminator(tf.keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.layer_list = [
          tf.keras.layers.Dense(20, activation="tanh"),
          tf.keras.layers.Dense(15, activation="tanh"),
          tf.keras.layers.Dense(1, activation="sigmoid"),
        ]

        self.metric_fake_loss = tf.keras.metrics.Mean(name="loss")
        self.metric_real_loss = tf.keras.metrics.Mean(name="loss")
        self.metric_loss = tf.keras.metrics.Mean(name="loss")

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    @tf.function
    def call(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return x
