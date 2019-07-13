# USAGE
# python detect_face_parts.py img.jpg 

# import the necessary packages
import sys
import numpy as np
import argparse
import imutils
import dlib
import cv2
from collections import OrderedDict 

debug = True
#Display image
def display(img, frameName="OpenCV Image"):
    if not debug:
        return
    h, w = img.shape[0:2]
    neww = 800
    newh = int(neww*(h/w))
    img = cv2.resize(img, (neww, newh))
    cv2.imshow(frameName, img)
    cv2.waitKey(0)


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)

	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords

def eyeAR(image, shape):
    clone = image.copy()

    #Left eyebrow augumentation
    #create list of marker points
    leftEyebrowList = []
    for (x, y) in shape[17:22]:
        leftEyebrowList.append((x,y))

    pts = np.array(leftEyebrowList)
    hull = cv2.convexHull(pts)
    mask = np.zeros(image.shape[0:2], np.uint8)
    cv2.drawContours(mask, [hull], -1, 255 ,-1)

    im2, c, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            
    if len(c) > 0:
        c = c[0]
        c = cv2.blur(c, (1, 7), 0)
        start = c[:, :, 0].argmin()
        end = c[:, :, 0].argmax()
        for idx in range(start, end):
            x, y = c[idx][0]
            tempList = []
            while True:
                if mask[y][x] == 255:
                    tempList.append((y,x))
                    y -= 1
                else:
                    break
            l = len(tempList)
            if l > 0:
                s = 30/l
            ct = 0
            for p in tempList:
                v = int(s*ct)
                clone[p[0]][p[1]] = [v, v, v]
                ct += 1
                        

    #Right eyebrow augumentation
    #create list of marker points
    RightEyebrowList = []
    for (x, y) in shape[22:27]:
        RightEyebrowList.append((x,y))

    pts = np.array(RightEyebrowList)
    hull = cv2.convexHull(pts)
    mask = np.zeros(image.shape[0:2], np.uint8)
    cv2.drawContours(mask, [hull], -1, 255 ,-1)

    #find contour
    im2, c, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            
    #draw the color shading
    if len(c) > 0:
        c = c[0]
        c = cv2.blur(c, (1, 7), 0)
        start = c[:, :, 0].argmin()
        end = c[:, :, 0].argmax()
        for idx in range(start, end):
            x, y = c[idx][0]
            tempList = []
            while True:
                if mask[y][x] == 255:
                    tempList.append((y,x))
                    y -= 1
                else:
                    break
            l = len(tempList)
            if l > 0:
                s = 30/l
            ct = 0
            for p in tempList:
                v = int(s*ct)
                clone[p[0]][p[1]] = [v, v, v]
                ct += 1
    return clone


def main(image=None):
    # load the input image, resize it, and convert it to grayscale
    #image = cv2.imread("example_01.jpg")
    #image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    clone = image.copy()
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        #Mouth augumentation
        upperLipList = []
        for (x, y) in shape[48:55]:
            upperLipList.append((x,y))
        upperLipList.append(shape[64])
        upperLipList.append(shape[63])
        upperLipList.append(shape[62])
        upperLipList.append(shape[61])
        upperLipList.append(shape[60])
	            
        pts = np.array(upperLipList)
            
        mask = np.zeros(image.shape[0:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, 255 ,-1)

        im2, c, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            
        if len(c) > 0:
            c = c[0]
            #smooth the contour curve
            c = cv2.blur(c, (1, 5), 0)
            start = c[:, :, 0].argmin()
            end = c[:, :, 0].argmax()
            for idx in range(start, end):
                x, y = c[idx][0]
                tempList = []
                while True:
                    if mask[y][x] == 255:
                        tempList.append((y,x))
                        y -= 1
                    else:
                        break
                l = len(tempList)
                if l > 0:
                    s = 150/l
                ct = l
                for p in tempList:
                    v = 255 - int(s*ct)
                    clone[p[0]][p[1]] = [v, 0, v]
                    ct -= 1

        #lower lip extraction
        #create list of marker points
        lowerLipList = []
        for (x, y) in shape[54:60]:
            lowerLipList.append((x,y))

        lowerLipList.append(shape[48])
        lowerLipList.append(shape[60])
        lowerLipList.append(shape[67])
        lowerLipList.append(shape[66])
        lowerLipList.append(shape[65])
        lowerLipList.append(shape[64])

        mask = np.zeros(image.shape[0:2], np.uint8)
        pts = np.array(lowerLipList)
        cv2.drawContours(mask, [pts], -1, 255 ,-1)
            
        im2, c, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

        #draw the color shading
        if len(c) > 0:
            c = c[0]
            #smooth the contour curve
            c = cv2.blur(c, (1, 5), 0)
            start = c[:, :, 0].argmin()
            end = c[:, :, 0].argmax()
            for idx in range(start, end):
                x, y = c[idx][0]
                tempList = []
                while True:
                    if mask[y][x] == 255:
                        tempList.append((y,x))
                        y -= 1
                    else:
                        break
                l = len(tempList)
                if l > 0:
                    s = 150/l
                ct = 0
                for p in tempList:
                    v = 255 - int(s*ct)
                    clone[p[0]][p[1]] = [v, 0, v]
                    ct += 1
        #Eyebrow augumentation 
        clone = eyeAR(clone, shape)
    
    blur = cv2.GaussianBlur(clone,(5,5),0)
    return blur
    


if __name__ == "__main__":
    # =====================================================
    # IMAGE LOADING
    # =====================================================
    if len(sys.argv) < 2:
        print("Invaid programme argument, Usage: python main.py <img_path>")
        sys.exit(1)

    path = sys.argv[1]
    if not path.endswith(".png") and not path.endswith(".jpg"):
        print("Must use a png or a jpg image to run the program.")
        sys.exit(1)

    #image = cv.imread(path)
    #in_file = os.path.join("D:\data", "image003.jpg")

    img = cv2.imread(path)
    if img is not None:
        print("Processing..... ", path)
        #apply augumentation
        res = main(img)
        display(res)
        sys.exit(1)
    else:
        print("Invalid input file.")
        sys.exit(1)
    