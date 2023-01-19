import matplotlib
from matplotlib import pyplot as plt
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import cv2 as cv

matplotlib.use('TkAgg')

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

# A class to compute kmeans clusters and generate a plot of the clusters
class FeaturePlot():
    
    # Class constructor - computes kmeans and maps them to image objects in dictionary
    def __init__(self, dic, number_of_clusters):
        
        self.dic = dic
        self.values_list = list(self.dic.values())
        
        self.number_of_clusters = number_of_clusters
        
        self.cluster_data = self.compute_kmeans(self.values_list, self.number_of_clusters)
        self.labels, self.centroids, self.stacked_float = self.cluster_data
        
        
        self.clustered_objects = []
        for i in range(self.number_of_clusters):
            self.clustered_objects.append(self.map_images(self.dic, i, self.labels, self.stacked_float))
        
    
    # Maps attributee value to image object
    def map_images(self,image_length_dict, cluster, labels, stacked_float):
        
        image_lengths = stacked_float[labels == cluster]
        
        images_in_same_cluster = [image for image, length in image_length_dict.items() if length in image_lengths]
        
        return images_in_same_cluster
        
    
    # Computes kmeans 
    def compute_kmeans(self, data, clusters, max_iterations=10, epsilon=1.0):
        stacked = np.hstack(data) # REMOVE IF DATA IS BEING PASSED AS VECTOR
        stacked = stacked.reshape((len(data), 1))  # Reshapes stacked array
        stacked_float = np.float32(stacked)
        
        # Define criteria = ( type, max-iter = 10, epsilon = 1.0)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, max_iterations, epsilon)
        
        # flags for avoiding code line breaks
        flags = cv.KMEANS_RANDOM_CENTERS
        
        # Applying KMeans
        attempts = max_iterations
        compactness, labels, centroids = cv.kmeans(stacked_float, clusters, None, criteria, attempts, flags)
        
        return labels, centroids, stacked_float
    
    # Generates plot representing attributes in clusters - attributes in the same cluster have the same colour
    def get_plot(self):
        
        labels, centroids, stacked_float = self.cluster_data
        
        fig = plt.figure()
        data = [[stacked_float[labels == cluster]]  for cluster in range(self.number_of_clusters)]
        data.sort(key=lambda x: x[0][0])
        
        plt.hlines(1, min(self.values_list), max(self.values_list), label=f'{min(self.values_list)}, {max(self.values_list)}')
        for i in range(len(data)):
            plt.eventplot(data[i], orientation='horizontal', colors=colors[i])
        
        # redraw the canvas
        fig.canvas.draw()
        
        # convert canvas to image
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))    
        
        plt.close(fig)
        return img
    
    # Returns the image objects in their clusters
    def get_clustered_objects(self):
        return self.clustered_objects
