import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris

#data
x_train = load_iris().data
y_train = load_iris().target
np.random.seed(100)
np.random.shuffle(x_train)
np.random.seed(100)
np.random.shuffle(y_train)
tf.random.set_seed(100)

#models.Sequential


class IrisModel(tf.keras.models.Model):
    def __init__(self):
        super(IrisModel, self).__init__()
        self.d1 = tf.keras.layers.Dense(3,activation='sigmoid',kernel_regularizer=tf.keras.regularizers.l2())

    def call(self,x):
        y = self.d1(x)
        return y


model = IrisModel()

#model.compile
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics =['sparse_categorical_accuracy'])

#model.fit
model.fit(x_train,y_train,batch_size=32,epochs=500,validation_split=0.2,validation_freq=20)
model.summary()


