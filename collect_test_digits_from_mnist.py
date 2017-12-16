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
    	