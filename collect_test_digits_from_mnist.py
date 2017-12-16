'''
由于自己手写的数字图像用来测试会得到很低的准确率，根据本人实验，觉得应该是图像风格所致，所以此py文件用于从mnist.test中生成用于测试的图像，再拿这些图像处理后测试
'''

import tensorflow.examples.tutorials.mnist.input_data as input_data
from PIL import Image



if __name__ == "__main__":
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    l = []
    for i in range(10):
    	for j in range(1000):
    		if mnist.test.labels[j][i]:
    			l.append(j)
    			break
    print(l)
    for i in range(10):
    	loc = l[i]
    	img = mnist.test.images[loc]
    	img = 255 - img * 255
    	img = img.astype("uint8")
    	img.resize((28, 28))
    	img = Image.fromarray(img).resize((400, 400), Image.ANTIALIAS)
    	img.save("test_digits" + str(i) + ".png")
    	