#  Clothes color detection using Computer vision and Machine Learning.

If a person is wearing a specific color, this system recognizes him and draws a rectangle around him (user specified color).

To understand the program, the programmer should have a basic knowledge of color spaces (mainly RGB and HSV) and of convolutional neural networks.

The program is written in python3. The main libraries used are OpenCV, scikit-learn, NumPy and matplotlib.

This program's concept is as follows:
1.	Scan the image and check if a person has been detected.
2.	Before creating the rectangle around the person, we must first ensure that the person is dressed in the correct color that the user is searching for. Because most images, particularly videos, contain noise (e.g., a single pixel of that color) that will lead to inaccurate results. This is why we use K-mean clustering to find the most dominant colors within the rectangle (which should delimit the individual).
3.	After finding the dominant colors, we should loop over them one by one, and check if there is the color we are looking for. If the color is found, then we proceed to draw the rectangle.

The trick behind the accuracy of this program is choosing the right HSV color lower and upper bound. Each color has its own range in HSV. The boundaries can be discovered by trial and error. Enter color picker into Google and try for yourself. We tried 3 colors and their range (blue, green and yellow), but you can try by yourself any other color of choice, just make sure to find the right upper and lower bounds. 

The main program to check is CCDv2.py in the Final codes folder. The other codes are some basic codes that were relevant to the main one.

This program was not 100% accurate. You should still adjust the value of the cluster or even the lower and higher boundaries for the color from time to time. Remember that executing additional clusters requires more time, making the application slower. Furthermore, if the video is busy and the software captures numerous individuals, it will be slower than capturing a single person (more people = more computations per frame = slower program).

Make sure to download the pre-trained caffe model before running the program: https://github.com/PINTO0309/MobileNet-SSD-RealSense/blob/master/caffemodel/MobileNetSSD/MobileNetSSD_deploy.caffemodel

Resources:
https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/
https://www.pyimagesearch.com/2014/05/26/opencv-python-k-means-color-clustering/
