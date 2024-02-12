import tensorflow as tf

class Discriminator(tf.keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.layer_list = [
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LeakyReLU(),
            
            tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ]

        self.metric_fake_loss = tf.keras.metrics.Mean(name="fake_loss")
        self.metric_real_loss = tf.keras.metrics.Mean(name="real_loss")
        self.metric_loss = tf.keras.metrics.Mean(name="loss")

        self.metric_real_accuracy = tf.keras.metrics.Accuracy(name="real_accuracy")
        self.metric_fake_accuracy = tf.keras.metrics.Accuracy(name="fake_accuracy")

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    @tf.function
    def call(self, x, training=False):
        for layer in self.layer_list:
            if isinstance(layer, tf.keras.layers.BatchNormalization) or isinstance(layer, tf.keras.layers.Dropout):
                x = layer(x, training)
            else:
                x = layer(x)
        return x
