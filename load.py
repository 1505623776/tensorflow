import tensorflow as tf
from PIL import Image
(train_data, train_target),(test_data, test_target) = tf.keras.datasets.mnist.load_data()
path = '/Users/huangrenming/Desktop/data/'
text = ''
for i in range(6000):
    print('saving image %d' % (i+1))
    im = Image.fromarray(train_data[i])
    im.save(path+'JPGE/%d.jpg' % (i+1))
    text += '%d.jpg  %d\n' % (i+1, train_target[i])

with open(path + 'text.txt','w') as file:
    file.write(text)
