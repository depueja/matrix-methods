import numpy as np 
from numpy.linalg import svd 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#load an image
image_path = r'INSERT IMAGE HERE'
im = plt.imread(image_path)

#display the image as check
#   plt.imshow(im)
#   plt.show()

#convert to np array
a = np.array(im)
#now, make it greyscale by removing third axis ("of color")
b = np.mean(a, 2)
print(f"original matrix size b4 SVD: {b.shape}")
#can see shape of image array: print(a.shape)

#perform SVD:
u, s, vt = np.linalg.svd(b, full_matrices=False)

#now, slice first k singular values to compress image
k = int(input(f"how many singular values would you like to take? you have {s.shape[0]} avaliable. "))
u1 = u[:,:k]
s1 = s[:k]
v1 = vt[0:k,:]
#recombine decomposed matrices into compressed matrix
compressed = u1 @ (np.diag(s1) @ v1)
print(f"matrix size after SVD and recombine: {compressed.shape}.")

plt.imshow(compressed, cmap='grey')
plt.axis('off')
plt.show()
