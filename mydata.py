import numpy as np
import matplotlib.pyplot
import imageio
from neural_network import n
img_array = imageio.imread("D:/Pycharmproject/minst_number/mydata_1.png", as_gray=True)
img_arr
img_data = 255.0 - img_array.reshape(784)
img_data = (img_data / 255.0 * 0.99) + 0.01
print(np.min(img_data))
print(np.max(img_data))
record = np.append(2, img_data)
matplotlib.pyplot.imshow(record[1:].reshape(28,28), cmap='Greys', interpolation='None')
correct_label = record[0]
inputs = record[1:]
outputs = n.query(inputs)
label = np.argmax(outputs)
print("network says:", label)
if (label == correct_label):
    print("match")
else:
    print("no match")
    pass

