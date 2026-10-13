import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from time import perf_counter
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--detector", help="Select between sift and orb.")
parser.parse_args()

valid_detectors = ("sift", "orb")
IMG_PATH = "f1.jpg"

def save_image(img_name, img):
    cv2.imwrite(img_name, img)

def show_image(img):
    figure(num=None,figsize=(100,100),dpi=80,facecolor='w',edgecolor='k')
    plt.imshow(img)
    plt.show()
    
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        return img
    else:
        raise FileNotFoundError("Invalid image file path. Check the path again.")
        return None

def detect_features(img, detection_method):
    if detection_method == "orb":
        detector = cv2.ORB_create()
    elif detection_method == "sift":
        detector = cv2.SIFT_create()
    else:
         raise Exception("Please select a method from orb and sift.")
    
    start = perf_counter()
    keypoints, descriptors = detector.detectAndCompute(img, None)
    end = perf_counter()
    print(f"Time taken by {detection_method} is {end-start}")
    return keypoints, descriptors

def main():
    detector = parser.parse_args().detector
    if detector not in valid_detectors:
        raise Exception("Please select between sift or orb.")
    img1 = load_image(IMG_PATH)
    keypoints1, descriptors1 = detect_features(img1, detector)
    #show_image(output_image)
    img2 = img1.copy()
    img2 = img2[1200: 1500, 1500:2000]
    keypoints2, descriptors2 = detect_features(img2, detector)
    print("Number of keypoints in image 1: ", len(keypoints1))
    print("Number of descriptors in image 1: ", len(descriptors1))    
    print("Number of keypoints in image 2: ", len(keypoints2))
    print("Number of descriptors in image 2: ", len(descriptors2))
    if detector=="orb":
        norm = cv2.NORM_HAMMING
    else:
        norm = cv2.NORM_L2
    matcher = cv2.BFMatcher(normType=norm, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2, None)
    matches_to_display = matches[:300] + matches[-300:]
    matched_image = cv2.drawMatches(img1,keypoints1,img2,keypoints2,matches_to_display,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    show_image(matched_image)

    ## Filtering good matches
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for m,n in matches:
        if m.distance < 0.20*n.distance:
            good_matches.append([m])
    
    matched_image = cv2.drawMatchesKnn(img1,keypoints1,img2,keypoints2,good_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #show_image(matched_image)
    

if __name__=="__main__":
    main()