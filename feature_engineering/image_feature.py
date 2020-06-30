import skimage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cgs
from skimage import io
#opencv tensorflow

# 图像shape
cat = io.imread('./datasets/cat.png')
dog = io.imread('./datasets/dog.png')
df = pd.DataFrame(['Cat', 'Dog'], columns=['Image'])


print(cat.shape, dog.shape)

print(cat)   #0-255,越小的值代表越暗，越大的值越亮

#coffee = skimage.transform.resize(coffee, (300, 451), mode='reflect')
fig = plt.figure(figsize = (8,4))
ax1 = fig.add_subplot(1,2, 1)
ax1.imshow(cat)
ax2 = fig.add_subplot(1,2, 2)
ax2.imshow(dog)

dog_r = dog.copy() # Red Channel
dog_r[:,:,1] = dog_r[:,:,2] = 0 # set G,B pixels = 0
dog_g = dog.copy() # Green Channel
dog_g[:,:,0] = dog_r[:,:,2] = 0 # set R,B pixels = 0
dog_b = dog.copy() # Blue Channel
dog_b[:,:,0] = dog_b[:,:,1] = 0 # set R,G pixels = 0

plot_image = np.concatenate((dog_r, dog_g, dog_b), axis=1)
plt.figure(figsize = (10,4))
plt.imshow(plot_image)

print(dog_r)

fig = plt.figure(figsize = (8,4))
ax1 = fig.add_subplot(2,2, 1)
ax1.imshow(cgs, cmap="gray")
ax2 = fig.add_subplot(2,2, 2)
ax2.imshow(dgs, cmap='gray')