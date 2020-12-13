import tensorflow as tf 
import numpy as np
import math

x = np.random.RandomState().rand(32, 2)
print(x)
y_ = [[x1+x2+(np.random.RandomState().rand()/10+0.05)] for (x1,x2) in x]
x = tf.cast(x, dtype=tf.float32)
w1 = tf.Variable(tf.random.normal([2, 1], stddev=1))
b1 = tf.Variable(tf.constant([1]))

epoch = 1500
Ir = 0.01

for epoch in range(epoch):
    with tf.GradientTape() as tape:
        y = tf.matmul(x, w1)
        loss_mes = tf.reduce_sum(tf.where(tf.greater(y, y_), 2*(y-y_), 3*(y_-y)))
    grads = tape.gradient(loss_mes, w1)
    w1.assign_sub(Ir * grads)
    # b1.assign_sub(Ir * grads[1])

    if epoch % 500 == 0:
        print("After %d training steps w1 is:" %epoch)
        print(w1.numpy())

print("Final w1 is :", w1.numpy())

x_ = [[100, 200]]
x_ = tf.cast(x_, dtype = tf.float32)
out = tf.matmul(x_, w1)
print(out)
