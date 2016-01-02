import numpy as np

a = np.zeros((2,17))
b = np.array([ 0,  0, 0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0])

a[0] = b
for l in a:
	print l

