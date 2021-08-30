import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import time as t

# read the image
# image = cv2.imread(r'C:\Users\Patss\Desktop\Internship\blueDetect2.jpg')
image = cv2.imread(r'C:\Users\Patss\Desktop\Internship\yellow2.jpg')
# get the number of clusters from the user
clusters = int(input('Enter the number of clusters: '))
cv2.imshow('image', image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)))
# reshape the image to be a list of pixels
image = image.reshape((image.shape[0] * image.shape[1], 3))


# check the link in the readMe file for more information about the helper methods

def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)  # Return evenly spaced values within a given interval.

    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    # return the histogram
    return hist


def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0
    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar


# cluster the pixel intensities
clt = KMeans(n_clusters=clusters)
t0 = t.time()
clt.fit(image)
# build a histogram of clusters and then create a figure
# representing the number of pixels labeled to each color
hist = centroid_histogram(clt)
values = []
numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)

for (percent, color) in zip(hist, clt.cluster_centers_):
    values.append([round(percent, 4), color])

# sorting the colors according to their percentages
values = sorted(values, key=lambda l: l[0], reverse=True)

t1 = t.time()
print("Clusterization took %0.5f seconds" % (t1 - t0))

cv2.waitKey(0)
# # print(values)
bar = plot_colors(hist, clt.cluster_centers_)
# show our color bart
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()
