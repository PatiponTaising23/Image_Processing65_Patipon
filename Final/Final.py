import cv2
import numpy as np
import scipy
import matplotlib.pylab as plt
from skimage import io
img = cv2.imread("Final\Template-1.png", 0)  
cap = cv2.VideoCapture("Final\left_output-1.avi", 0)


sift = cv2.xfeatures2d.SIFT_create()
kp_image, desc_image = sift.detectAndCompute(img, None)
fl = cv2.flann
bf = cv2.BFMatcher


index_params = dict(algorithm=0, trees=5)
search_params = dict()
bf = cv2.FlannBasedMatcher(index_params, search_params)


while cap.isOpened() :


    ret, frame = cap.read()
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)
    matches = bf.knnMatch(desc_image, desc_grayframe, k=2)
    good_points = []


    for m, n in matches:
        if m.distance < 0.79999 * n.distance:
            good_points.append(m)


            
    img3 = cv2.drawMatches(img, kp_image, grayframe, kp_grayframe, good_points, grayframe)


    if len(good_points) > 10:


        query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
        matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 0.5)
        matches_mask = mask.ravel().tolist()



        h, w = img.shape
        point = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dest = cv2.perspectiveTransform(point, matrix)
        LASTDETECT = cv2.polylines(frame, [np.int32(dest)], True, (0, 255, 0), 3)
        cv2.imshow("Homography", LASTDETECT)

        
    else:
        cv2.imshow("Homography", grayframe)
        #cv2.imshow("Image", img)
        #cv2.imshow("grayFrame", grayframe)
        #cv2.imshow("img3", img3)
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()