#Phase 1 — Load & Explore the Data
    #Get familiar with the shape of the dataset. Understand what a single data point looks like, and how to render it as an image.
#Phase 2 — Preprocessing
    #Prepare the data matrix so it's in the right form for PCA. There's one step here that's easy to forget but matters a lot.
#Phase 3 — The Decomposition
    #Implement PCA from scratch using what you already know. You'll make a decision here about how to compute it, and there are two valid approaches with different tradeoffs.
#Phase 4 — The Scree Plot
    #Before projecting anything, interrogate your results. How much of the data's variance lives in each component? How many do you actually need?
#Phase 5 — 2D Projection & Scatter Plot
    #Project the full dataset down to 2 dimensions and visualize it, colored by digit class. This is the payoff moment.
#Phase 6 — Eigendigits
    #Visualize the principal components themselves as images. Try to interpret what each one "means."
#Phase 7 — Reconstruction
    #Pick a few digits and reconstruct them using an increasing number of components. Connect this back to what you did with image compression.
#! Stretch goal — an interactive slider that lets you explore the projection or reconstruction live.


#----------------------------------------------------------------------------------------------------------


#important libraries
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import fetch_openml
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

#load data
mnist = fetch_openml(name='mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target # Features and labels

#DATA SANITY CHECKS
    #print(X.shape)
    #print(X[:5]) <- should print only zeros, because edges are blank/black padding, so no actual 'image data'
    #plt.imshow(X[0].reshape(28, 28), cmap='grey')
    #plt.show()

#START PCA
#center to standardize data; usually, have to center AND scale, but here, mnist is already scaled (as every pixel on same 0-255 scale)

n=5 #controls the amount of random samples we want!
data_array_samples = np.concatenate([
    np.random.choice(np.where(y==str(d))[0], n, replace = False)
    for d in range(10)
])

#index the original data with the positions of the samples

X_sample = X[data_array_samples]

X_centered = X - X.mean(axis=0)
X_centered_sample = X_sample - X_sample.mean(axis=0)

X_cov = X_centered.T @ X_centered / 70000
X_cov_sample = X_centered_sample.T @ X_centered_sample / n

eig_vals, eig_vecs = np.linalg.eigh(X_cov)

eigs_descending_index = np.flip(np.argsort(eig_vals))

eig_vals_sorted_descending = eig_vals[eigs_descending_index]
eig_vecs_sorted_descending = eig_vecs[:, eigs_descending_index]


#----------------------------------------------------------------------------------------------------------
#covariance matrix has now been calculated. the formula states:
    #X_cov = E[(Xi-E[Xi])(Xj-E[Xj])]
#in other words, the expectation (mean) of the average across i pixels TIMES the expectation of the average across j pixels
#then, in code, this is just the X_centered matrix, which is our data averaged over all the columns, i.e., per pixel.
#the reason we do X_centered.T @ X_centered is becasue this gives us that i * j multiplication, and the result is 
#784 x 784, which is number of pixel pairs. for i and j specifically, "When pixel 300 is bright, is pixel 400 also bright?" → entry (300, 400)
#hence the need for 2 indices.

#np.argsort generally returns in ascending order, so use np.flip to reverse to descending order.
#then, since eigenvalues are just 1D array, can index directly.
#however, for vectors, its 2D array, so have to index columns, because eigenvectors stored as columns, not rows
# i.e., eigenvector's information is in a column, so to represent it fully need all elements of a given column
#----------------------------------------------------------------------------------------------------------

variance_proportions = eig_vals_sorted_descending / np.sum(eig_vals_sorted_descending)

#plt.plot(np.cumsum(variance_proportions))
#plt.show()
#doing above results in 136 eigenvalues capturing 95% of variance. so, k = 136 (will make change-able later!)
# now project to 2D

X_projected_sample = X_centered_sample @ eig_vecs_sorted_descending[:, 0:3]

#print(X_projected.shape) to confirm that shape is (70000,3), EDIT: 0:3 makes possible 3D scatter plot 
# so we have 700000 images represented as a 3-vector in eigenbasis
# i.e., how much of principle component 1,2, and 3 affect image.
scatter_3d = plt.figure()
axis = scatter_3d.add_subplot(111, projection='3d')

y_sample = y[data_array_samples]
scatter = axis.scatter(X_projected_sample[:,0], X_projected_sample[:,1], X_projected_sample[:,2], c=y_sample.astype(int), cmap='tab10', s=200, alpha=0.5, picker = True)
plt.colorbar(scatter)
#make points clickable
def on_click(event):
    ind = event.ind[0]  # index of clicked point
    x = X_projected_sample[ind, 0]
    y_coord = X_projected_sample[ind, 1]
    z = X_projected_sample[ind, 2]
    label = y_sample[ind]
    print(f"Digit: {label}, x: {x:.2f}, y: {y_coord:.2f}, z: {z:.2f}")
scatter_3d.canvas.mpl_connect('pick_event', on_click)

#---------------------------------------------------------------------------#

#now, show eigenlines for the components as a seperate figure
figure, axes = plt.subplots(4,4)

#loop over eigenvectors
for i in range(16):
    axes[i//4, i % 4].imshow(eig_vecs_sorted_descending[:, i].reshape(28, 28), cmap='grey')


#---------------------------------------------------------------------------#
#fills out the 4x4 subplots grid with largest eigenvectors in top left, then descending to the right
# called 'eigendigits!'
#here's how to read the 'heatmap'
###White — this pixel varies strongly positively along this component. Images with a high score on this component tend to be bright here.
###Black — this pixel varies strongly negatively. Images with a high score tend to be dark here, and vice versa.
###Gray — this pixel contributes almost nothing to this component. It barely varies along this direction at all.

####  Bright region in eigendigit → having ink here correlates with a high (positive) score on this component
####  Dark region in eigendigit → having ink here correlates with a low (negative) score on this component

#each dot's position on the scatter plot is its coordinate along principal component 1 (x axis) and principal component 2 (y axis)

# a digit's correlation with the first eigendigit tells you a lot about its broad structural shape (round vs straight), 
# which is highly informative but not the whole story. Combining correlations across all 136 components gives you a much richer, 
# more complete description — which is why the full projection to 136 dimensions works so well for downstream tasks.
#---------------------------------------------------------------------------#


#now to reconstruct some digits!
#we know that x projected = X_centered @ Vk, where Vk is number of eigenvalues. MAKE THIS USER INPUT-BASED LATER!!!!!
#k_values = [1, 5, 10, 30, 136]

#fig, axes = plt.subplots(1, 6, figsize=(15, 3))

#for idx, k in enumerate(k_values):
#    V_k = eig_vecs_sorted_descending[:, :k]
#    X_projected_k = X_centered @ V_k
#    X_reconstructed = (X_projected_k @ V_k.T) + X.mean(axis=0)
#    axes[idx].imshow(X_reconstructed[0].reshape(28, 28), cmap='gray')
#    axes[idx].set_title(f'k={k}')
#    axes[idx].axis('off')

# original for comparison
#axes[5].imshow(X[0].reshape(28, 28), cmap='gray')
#axes[5].set_title('original')
#axes[5].axis('off')

#plt.show()

#make the k-values a slider! and compare with original (comparison written by Claude)
fig, (ax_image, ax_original) = plt.subplots(1, 2)
plt.subplots_adjust(bottom=0.25)
# original (never changes)
ax_original.imshow(X[0].reshape(28, 28), cmap='gray')
ax_original.set_title('Original')
ax_original.axis('off')


ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
k_slider = Slider(
    ax=ax_slider,
    label='# EDig. Approx',
    valmin=0,
    valmax=136,
    valinit=1,
)
### valmax shows us how many eigendigits to approximate image with.


#initial draw
V_k = eig_vecs_sorted_descending[:, :1]
X_projected_k = X_centered @ V_k
X_reconstructed = (X_projected_k @ V_k.T) + X.mean(axis=0)
img = ax_image.imshow(X_reconstructed[0].reshape(28, 28), cmap='gray')

#update draw with slider funciton
def update(val):
    k = int(k_slider.val)
    V_k = eig_vecs_sorted_descending[:, :k]
    X_projected_k = X_centered @ V_k
    X_reconstructed = (X_projected_k @ V_k.T) + X.mean(axis=0)
    img.set_data(X_reconstructed[0].reshape(28, 28))
    fig.canvas.draw_idle()

k_slider.on_changed(update)
plt.show()

#---------------------------------------------------------------------------#
#Each image is being expressed as a linear combination of eigenvectors, just like expressing a vector in a new basis. 
#The coefficients are the projection coordinates, and the eigenvectors are the new basis vectors.
#You're adding up 784-dimensional vectors, scaled by their coefficients. The result is also a 784-dimensional vector — one number per pixel. 
#That's already an image, just flattened. Then .reshape(28, 28) folds it back into a 2D grid and imshow displays it.

## Linear combination → produces a 784-dimensional vector of numbers
## .reshape(28, 28) → arranges those numbers into a 2D grid
## imshow → hands that grid to matplotlib
## cmap='gray' → tells matplotlib "map these numbers to a brightness scale"
## Monitor → physically lights up each pixel at the corresponding brightness
#---------------------------------------------------------------------------#

#and we're done!! only final thing to note is that we don't actually get a 'perfect' approximation when we use
#max eigendigits, which means we add more eigenvectors to our basis ("how many of the most important eigendigits to include, in order of their eigenvalues."),
#as our matrix operations have rounding to the 15th decimal place via float point convention, and np.linalg.eigh actually has small
#numerical errors as it is an approximation, under the hood!