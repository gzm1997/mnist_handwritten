import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
from PIL import Image
import numpy
import cv2

#从mnist数据集中加载训练和测试的数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#x用来存放图像，维度为[None, 784]，None意味着图像数量不限定，784表示每张图表示成784个点的一维向量
x = tf.placeholder(tf.float32, [None, 784])
#线性回归的参数W，维度为[784, 10]
W = tf.Variable(tf.zeros([784,10]))
#线性回归的参数b，维度为[10]
b = tf.Variable(tf.zeros([10]))
#让W和x矩阵相乘，加上b，然后计算softmax，softmax(x)=normalize(exp(x))
y = tf.nn.softmax(tf.matmul(x,W) + b)
#y_为用来存放正确的y结果
y_ = tf.placeholder("float", [None,10])
#使用交叉熵，以在后面使用梯度下降法根据交叉熵确定最佳的W和b
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#使用梯度下降法最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#初始化所有变量
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#训练1000次
for i in range(1000):
	#每次拿100张图像和标签来训练，batch_xs为训练的图像，batch_ys为对应的标签
    batch_xs, batch_ys = mnist.train.next_batch(100)
    #根据最小梯度下降法降低交叉熵
    sess.run(train_step, {x: batch_xs, y_: batch_ys})

#correct_prediction表示下面的预测是否正确
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#accuracy表示准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#使用mnist.test里面的数据进行测试，计算准确率
print("accuracy", sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels}))


#拿文件夹test_digits里面的0-9图片进行测试
for i in range(10):
	#读取图像为灰度图
	i = Image.open("test_digits/" + str(i) + ".png").convert("L")
	#缩小为28*28
	i = i.resize((28, 28), Image.ANTIALIAS)
	#获取灰度矩阵
	arr = numpy.array(i, dtype = "float32")
	#图片二值化
	ret, arr = cv2.threshold(255 - arr, 90, 255, cv2.THRESH_BINARY)
	#像素值归一化和展平为一维向量
	arr = (arr / 255).flatten()
	#进行测试，输出预测结果
	narr = numpy.zeros((1, 784))
	narr[0] = arr
	y_r = sess.run(y, {x: narr}) 	
	print(sess.run(tf.argmax(y_r, 1)))






