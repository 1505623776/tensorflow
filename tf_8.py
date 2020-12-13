import tensorflow as tf
import os
import numpy as np
x_train_save_path = '/Users/huangrenming/Desktop/tf_data/tf_8/x_train.npy'
x_test_save_path = '/Users/huangrenming/Desktop/tf_data/tf_8/x_test.npy'
y_train_save_path = '/Users/huangrenming/Desktop/tf_data/tf_8/y_train.npy'
y_test_save_path = '/Users/huangrenming/Desktop/tf_data/tf_8/y_test.npy'
if os.path.exists(x_train_save_path) and os.path.exists(x_test_save_path) and os.path.exists(y_train_save_path) and os.path.exists(y_test_save_path):
    print('----------Load Data----------')
    x_train_save = np.load(x_train_save_path)
    y_train = np.load(y_train_save_path)
    x_test_save = np.load(x_test_save_path)
    x_train = np.load(y_test_save_path)
    x_train = np.reshape(x_train_save, (len(x_train_save), 32, 32, 3))
    x_test = np.reshape(x_test_save, (len(x_test_save), 32, 32, 3))

else:
    print('---------Downloading Data----------')
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0
    # print('----------Saving Data----------')
    # x_train_save = np.array(np.reshape(x_train, (len(x_train), -1)))
    # x_test_save = np.array(np.reshape(x_test,(len(x_test), -1)))
    # np.save(x_train_save, x_train_save_path)
    # np.save(y_train, y_train_save_path)
    # np.save(x_test_save, x_test_save_path)
    # np.save(y_test, y_test_save_path)


class BaseLine(tf.keras.models.Model):
    def __init__(self):
        super(BaseLine, self).__init__()
        self.C = tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same')
        self.B = tf.keras.layers.BatchNormalization()
        self.A = tf.keras.layers.Activation('relu')
        self.P = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')
        self.D = tf.keras.layers.Dropout(0.2)
        self.flatten = tf.keras.layers.Flatten()
        self.f1 = tf.keras.layers.Dense(128, activation='relu')
        self.d2 = tf.keras.layers.Dropout(0.2)
        self.f2 = tf.keras.layers.Dense(10,  activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.C(inputs)
        x = self.B(x)
        x = self.A(x)
        x = self.P(x)
        x = self.D(x)
        x = self.flatten(x)
        x = self.f1(x)
        x = self.d2(x)
        y = self.f2(x)
        return y


model = BaseLine()
checkpoint_save_path = '/Users/huangrenming/Desktop/tf_data/tf_8/checkpoint/cifar10.ckpt'
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy']
              )
if os.path.exists(checkpoint_save_path+'.index'):
    print('----------Load Model----------')
    model.load_weights(checkpoint_save_path)


cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_save_path,
    save_weights_only=True,
    save_best_only=True,

)

history = model.fit(
    x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_steps=1,
    callbacks=[cp_callback]
)
model.summary()

