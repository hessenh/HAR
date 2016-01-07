import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib.image as mpimg
import pandas as pd 

PREDICTION = '../domain_knowledge/prediction.csv'
ACTUAL = '../predictions/original.csv'

df_prediction = pd.read_csv(PREDICTION, header=None, sep='\,', engine='python')
df_actual = pd.read_csv(ACTUAL, header=None, sep='\,', engine='python')


prediction = df_prediction.as_matrix()
actual = df_actual.as_matrix()

image_dict = {1:'sit.png',2:'stand.png',3:'stand.png',
			4:'stand.png',5:'stand.png',6:'sit.png',
			7:'stand.png',8:'stand.png',9:'stand.png'
             }
fig = plt.figure()
ax1=fig.add_subplot(1,2,1)
ax2=fig.add_subplot(1,2,2)

ims=[]
for i in range(0, 10):
	image = 'images/sit.png'
	im = ax1.imshow(mpimg.imread(image))
	image = 'images/stand.png'
	im2 = ax2.imshow(mpimg.imread(image))

	ims.append([im, im2])

def prediction_anim(i):
	label = prediction[i][0]
	print label
	image = 'images/'+image_dict.get(label)
	return plt.imshow(mpimg.imread(image)), plt.imshow(mpimg.imread(image))


anim = animation.FuncAnimation(fig, prediction_anim, interval=500)

plt.show()