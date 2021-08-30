from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np
import time as t
import colorsys
import cv2


def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    # return the histogram
    return hist


# lower and upper bound for the colors in HSV color space
color = str(input('What color do you want to detect: Blue/Green/Yellow'))

while True:
    if color.lower() == 'blue':
        lower = np.array([95, 100, 50])
        upper = np.array([125, 200, 200])
        break
    elif color.lower() == 'yellow':
        lower = np.array([19, 50, 50])
        upper = np.array([35, 255, 255])
        break
    elif color.lower() == 'green':
        lower = (35, 86, 86)
        upper = (75, 255, 255)
        break
    else:
        color = str(input('Wrong input! What color do you want to detect: Blue/Green/Red  '))

# set the number of clusters
clusters = int(input('Enter the number of clusters: '))

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# random color of the rectangle around the objects
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# make sure you downloaded the pre-trained model (from the link) and the txt file from the repo
net = cv2.dnn.readNetFromCaffe(r'C:\Users\Patss\Desktop\MobileNetSSD_deploy.prototxt.txt',
                               r"C:\Users\Patss\Downloads\MobileNetSSD_deploy.caffemodel")

# the machine should be at least this % confident that the detected object
# is a person before drawing a rectangle
conf = float(input('Please enter confidence: '))

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD implementation)

# insert your video
video = cv2.VideoCapture(r'C:\Users\Patss\Desktop\Internship\pov2.mp4')
# video = cv2.VideoCapture(r'C:\Users\Patss\Downloads\traffic2.mp4')

# process the video frame by frame
while True:
    ret, image = video.read()

    # if video ended, exit the loop so that the program doesn't crash
    if not ret:
        break

    image = cv2.resize(image, (720, 520))
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and predictions
    # [INFO] computing object detections...
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
        if confidence > conf:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # only proceed with color detection if the found object is a person
            if CLASSES[idx] == 'person':
                try:
                    # crop the person from the whole image
                    cropped = image[startY:endY, startX:endX]
                    # resize to 30x30 so that clustering the colors doesn't take too much time
                    # a 30x30 image took 0.03 seconds to cluster
                    cropped = cv2.resize(cropped, (30, 30))

                    # k-means to check to get dominant colors in the cropped image
                    clt = KMeans(n_clusters=clusters)
                    # reshape the image from a 3D array into a 2D array
                    cropped = cropped.reshape((cropped.shape[0] * cropped.shape[1], 3))

                    clt.fit(cropped)
                    # build a histogram of clusters and then create a figure
                    # representing the number of pixels labeled to each color
                    hist = centroid_histogram(clt)
                    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
                    # add the rgb values into a list
                    values = []

                    # storing the RGB color and its percentage in a list
                    for (percent, color) in zip(hist, clt.cluster_centers_):
                        values.append([round(percent, 4), color])

                    # loop over the list and for each color, check if it is the one we are looking for
                    for value in values:
                        # create a blank image
                        blank = np.zeros((10, 10, 3), np.uint8)
                        # make the blank image the same color as the color in the list
                        blank[:] = (int(value[1][2]), int(value[1][1]), int(value[1][0]))
                        # don't forget that OpenCV reads the values as BGR and not RGB
                        # switch to HSV for easier and better results for comparing
                        blank = cv2.cvtColor(blank, cv2.COLOR_BGR2HSV)
                        # threshold the image with the upper and lower bound previously chosen
                        thresh = cv2.inRange(blank, lower, upper)
                        # check if there is any white (non-zero) pixel in the image. If there is, then
                        # the color was found
                        count = np.sum(np.nonzero(thresh))
                        if count != 0:  # color is found
                            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                            print("[INFO] {}".format(label))
                            cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
                            y = startY - 15 if startY - 15 > 15 else startY + 15
                            cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                # this is in case the cv2.resize() function wont crash the code
                # this is due to the new frame trying to be processed before the previous one
                # is completely done with computations
                except Exception as e:
                    print(str(e))

    cv2.imshow("Output", image)
    # let the video automatically run and press 'q' on your keyboard to quit
    # the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
