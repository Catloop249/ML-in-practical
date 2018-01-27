import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#Pull data from minis
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

# Create tensor x as value input, 28x28 = 784 pixel -> varible x is vector of 784 input feature, we set none because we don't know length of data,number of row items
X= tf.placeholder(tf.float32,shape=[None,784])
# Create tensor y as predict probability of each digit 0 - 9 ex: [0.5 0 0.7 0 0.9 0 0 0 0 0]
Y = tf.placeholder(tf.float32,[None,10])
# Create inital weights and bias
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
# Create hypothesis function
hypo = tf.matmul(X,W)+b
# activate function - use softmax 
hypo_softmax = tf.nn.softmax(hypo)
# define cost function use cost entropy
cost_entropy= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=hypo_softmax))
# define optimizer
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cost_entropy)

#================

# Train Model - iterate minimize cost
#initialize varible model
init = tf.global_variables_initializer()
session=tf.Session()
session.run(init)
#perfom in 100 steps
for i in range(1000):
    batch_x,batch_y=mnist.train.next_batch(100) # get 100 random data x:image y is labels [0-9]
    session.run(train_step,feed_dict={X:batch_x,Y:batch_y})


# Evaluate Model - compare highest propability and actual digit
correct_predict = tf.equal(tf.argmax(Y,1),tf.argmax(hypo_softmax,1))
accuracy = tf.reduce_mean(tf.cast(correct_predict,tf.float32))
test_accuracy = session.run(accuracy,feed_dict= {X:mnist.test.images,Y:mnist.test.labels})
print("Test Accuray {0}%".format(test_accuracy*100))
session.close()