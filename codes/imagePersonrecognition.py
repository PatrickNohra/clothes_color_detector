import numpy as np
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


# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# color of the rectangle around the object
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# [INFO] loading model...
net = cv2.dnn.readNetFromCaffe(r'C:\Users\Patss\Desktop\MobileNetSSD_deploy.prototxt.txt',
                               r"C:\Users\Patss\Downloads\MobileNetSSD_deploy.caffemodel")

conf = float(input('Please enter confidence: '))

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD implementation)


# image = cv2.imread(r'C:\Users\Patss\Desktop\Internship\green1.jpg')
image = cv2.imread(r'C:\Users\Patss\Desktop\Internship\yellow2.jpg')
# video = cv2.VideoCapture(r'C:\Users\Patss\Downloads\traffic2.mp4')

# process the frame
image = cv2.resize(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)))
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
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            print("[INFO] {}".format(label))
            cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
cv2.imshow("Output", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
