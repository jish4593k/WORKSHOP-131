import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from scipy import misc
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

def display_image(image):
    plt.imshow(image)
    plt.show()

def k_means_clustering(image, num_clusters):
    # Reshape the image into an array (n,3), n = # of pixels, 3 for r,g,b
    pixels = image.reshape((-1, 3))

    
    pixels_tensor = torch.from_numpy(pixels).float()

    
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(pixels)
    
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    return centroids, labels

def compress_image(image, centroids, labels):
    
    compressed_pixels = centroids[labels, :]


    compressed_image = compressed_pixels.reshape(image.shape)

    return compressed_image

def main():
    print('Running K-Means clustering on pixels from an image.\n\n')

    image_name = 'sunset.png'

   
    A = misc.imread(image_name)
    row, column, rgb_rgba = A.shape

    display_image(A)

 
    A_new = A.reshape((row*column), rgb_rgba)


    num_clusters = 5

   g
    centroids, labels = k_means_clustering(A_new, num_clusters)

    print(min(labels), max(labels))

    print('\nApplying K-Means to perform image compression.\n\n')


    compressed_image = compress_image(A, centroids, labels)

    display_image(compressed_image)

if __name__ == "__main__":
    main()
