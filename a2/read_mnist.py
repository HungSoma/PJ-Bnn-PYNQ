from mlxtend.data import loadlocal_mnist

X, y = loadlocal_mnist(
        images_path='/home/rito/convert_mnist/a2/Convert-own-data-to-MNIST-format/convert_MNIST/train-images-idx3-ubyte', 
        labels_path='/home/rito/convert_mnist/a2/Convert-own-data-to-MNIST-format/convert_MNIST/train-labels-idx1-ubyte')

print('Dimensions: %s x %s' % (X.shape[0], X.shape[1]))
print('\n1st row', X[0])

import numpy as np

print('Digits:  0 1 2 3 4 5 6 7 8 9')
print('labels: %s' % np.unique(y))
print('Class distribution: %s' % np.bincount(y))