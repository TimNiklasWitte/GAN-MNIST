import tensorflow as tf

class Generator(tf.keras.Model):

    def __init__(self):
        super(Generator, self).__init__()

        self.layer_list = [
          tf.keras.layers.Dense(10, activation="tanh"),
          tf.keras.layers.Dense(15, activation="tanh"),
          tf.keras.layers.Dense(20, activation="tanh"),
          tf.keras.layers.Dense(28*28, activation="tanh")
        ]

        self.metric_loss = tf.keras.metrics.Mean(name="loss")
     
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        self.noise_dim = 50

    @tf.function
    def call(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return x

