import tensorflow as tf
import numpy as np
import os
from matplotlib import pyplot as plt

input_word = 'abcde'
w_to_id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
id_to_onehot = {
    0: [1., 0, 0, 0, 0], 1: [0, 1., 0, 0, 0], 2: [0, 0, 1., 0, 0],
    3: [0, 0, 0, 1., 0], 4: [0, 0, 0, 0, 1.]
}
x_train = [id_to_onehot[w_to_id['a']],
           id_to_onehot[w_to_id['b']],
           id_to_onehot[w_to_id['c']],
           id_to_onehot[w_to_id['d']],
           id_to_onehot[w_to_id['e']]
           ]
y_train = [w_to_id['b'],
           w_to_id['c'],
           w_to_id['d'],
           w_to_id['e'],
           w_to_id['a']]
np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)

x_train = np.reshape(x_train, (len(x_train),1,5))
y_train = np.array(y_train)

model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(3),
    tf.keras.layers.Dense(5, activation='softmax')
])
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(0.01),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
#     metrics=['sparse_categorical_accuracy']
# )
checkpoint_save_path = '/Users/huangrenming/Desktop/tf_data/checkpoint/rnn_onehot_1pre1.ckpt'

if os.path.exists(checkpoint_save_path + '.index'):
    print('----------Load Model----------')
    model.load_weights(checkpoint_save_path)

# cp_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_save_path,
#     save_best_only=True, save_weights_only=True, monitor='loss'
# )
# history = model.fit(
#     x_train, y_train, batch_size=1, epochs=150, callbacks=[cp_callback]
# )
# model.summary()
while True:
    alphabet1 = input('input test alphabet:')
    res = ''
    for i in alphabet1:
        alphabet = [id_to_onehot[w_to_id[i]]]
        alphabet = np.reshape(alphabet, (1, 1, 5))
        result = model.predict(alphabet)
        pred = tf.argmax(result, axis=1)
        pred = int(pred)
        res += input_word[pred]
    tf.print(alphabet1 + '->' + res)

