import tensorflow as tf
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import os
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
train_path = '/Users/huangrenming/Desktop/data/JPGE/'
train_text = '/Users/huangrenming/Desktop/data/text.txt'


x_train_savepath = '/Users/huangrenming/Desktop/data/x_train.npy'
y_train_savepath = '/Users/huangrenming/Desktop/data/y_train.npy'
x_test_savepath = '/Users/huangrenming/Desktop/data/x_test.npy'
y_test_savepath = '/Users/huangrenming/Desktop/data/y_test.npy'


def generateds(path, txt):
    f = open(txt, 'r')
    contents = f.readlines()
    f.close()
    x, y = [], []
    for content in contents:
        value = content.split()
        img_path = path + value[0]
        img = Image.open(img_path)
        img = np.array(img.convert('L'))
        img = img/255.
        x.append(img)
        y.append(np.int(value[1]))
        print('loading:' + content)

    x = np.array(x)
    y = np.array(y)
    y = y.astype(np.int)
    return (x[:-1000], y[:-1000]),(x[-1000:], y[-1000:])


if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath) and os.path.exists(x_test_savepath)\
        and os.path.exists(y_test_savepath):
    print('---------Load Datasets---------')
    x_train_save = np.load(x_train_savepath)
    x_test_save = np.load(x_test_savepath)
    y_train = np.load(y_train_savepath)
    y_test = np.load(y_test_savepath)
    x_train = np.reshape(x_train_save, (len(x_train_save), 28, 28))
    x_test = np.reshape(x_test_save, (len(x_test_save), 28, 28))
else:
    print('---------Generateds Datasets---------')
    (x_train, y_train),(x_test, y_test) = generateds(train_path, train_text)

    print('----------Save Datasets----------')
    x_train_save = np.reshape(x_train, (len(x_train), -1))
    x_test_save = np.reshape(x_test,(len(x_test),-1))
    np.save(x_train_savepath, x_train_save)
    np.save(y_train_savepath, y_train)
    np.save(x_test_savepath, x_test_save)
    np.save(y_test_savepath, y_test)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['sparse_categorical_accuracy']
)
checkpoint_save_path = '/Users/huangrenming/Desktop/data/checkpoint/mnist.ckpt'

if os.path.exists(checkpoint_save_path+'.index'):
    print('----------Load The Model---------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_save_path,
    save_weights_only=True,
    save_best_only=True
)

history = model.fit(
    x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test,y_test),validation_steps=1,
    callbacks=[cp_callback])
model.summary()
# np.set_printoptions(threshold=np.inf)
# print(model.trainable_variables)
# with open('/Users/huangrenming/Desktop/data/weigtehs.txt','w' )as file:
#     for v in model.trainable_variables:
#         file.write(str(v.name)+'\n')
#         file.write(str(v.shape)+'\n')
#         file.write(str(v.numpy())+'\n')
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy ')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
