import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from time import perf_counter

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
        detector = cv2.ORB_create(nfeatures=11298)
    elif detection_method == "sift":
        detector = cv2.SIFT_create()
    else:
         raise Exception("Please select a method from orb and sift.")
    
    start = perf_counter()
    features = detector.detect(img)
    end = perf_counter()
    print(f"Time taken by {detection_method} is {end-start}")
    return features, end-start

def main():
    img = load_image(IMG_PATH)
    print("Image shape: ", img.shape)
    keypoints, time_taken_by_detector = detect_features(img, "orb")
    print("Number of keypoints: ", len(keypoints))
    output_image = cv2.drawKeypoints(img, keypoints, 0, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)    
    #show_image(output_image)
    save_image("image_with_keypoints_orb.jpg", output_image)
    
if __name__=="__main__":
    main()