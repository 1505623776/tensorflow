import tensorflow as tf
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt 
import numpy as np
import  time
_data =load_iris().data
_table = load_iris().target

np.random.seed(11)
np.random.shuffle(_data)
np.random.seed(11)
np.random.shuffle(_table)
tf.random.set_seed(11)

train_data = _data[:-30]
train_table = _table[:-30]
test_data = _data[-30:]
test_table = _table[-30:]

train_data = tf.cast(train_data,tf.float32)
test_data  = tf.cast(test_data,tf.float32)

train_db = tf.data.Dataset.from_tensor_slices((train_data,train_table)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((test_data,test_table)).batch(32)

w = tf.Variable(tf.random.truncated_normal([4,3],stddev = 0.1))
b = tf.Variable(tf.random.truncated_normal([3],stddev = 0.1))
Ir = 0.1
m_w, m_b = 0, 0
train_loss_results = []
test_acc = []
epoch = 500
loss_all = 0
now_time = time.time()
for epoch in range(epoch):
    for step,(train_data,train_table) in enumerate(train_db):
        with tf.GradientTape() as tape:
            
            y = tf.matmul(train_data,w)+b
            y = tf.nn.softmax(y)
            y_ = tf.one_hot(train_table,depth = 3)
            loss = tf.reduce_mean(tf.square(y_-y))
            loss_all += loss.numpy()
        grads = tape.gradient(loss,[w,b])
##################################
        #adagrad
        m_w += tf.square(grads[0])
        m_b += tf.square(grads[1])
        w.assign_sub(Ir*grads[0]/tf.sqrt(m_w))
        b.assign_sub(Ir*grads[1]/tf.sqrt(m_b))
###############################
    # print("Epoch{},loss:{}".format(epoch,loss_all/4))
    train_loss_results.append(loss_all/4)
    loss_all = 0
    total_correct,total_number = 0,0
    for test_data,test_table in test_db:
        
        y = tf.matmul(test_data,w)+b
        y = tf.nn.softmax(y)
        pred = tf.argmax(y,axis=1)
        pred = tf.cast(pred,test_table.dtype)
        correct = tf.cast(tf.equal(pred,test_table),tf.int32)
        correct = tf.reduce_sum(correct)
        total_correct +=int(correct) 
        total_number += test_data.shape[0]

    acc = total_correct/total_number
    test_acc.append(acc)
    # print("Test-acc:",acc)
    # print("-----------------")
print("耗时：%d" %(time.time()-now_time))
plt.title('Loss Function Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(train_loss_results,label = "$Loss$")
plt.legend()
plt.show()

plt.title('ACC Function Curve')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.plot(test_acc,label = "$ACC$")
plt.legend()
plt.show()
