import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True, help="Input image")
ap.add_argument("-c", "--cascade", type=str, help="haar cascade file")
args =vars(ap.parse_args())

print("[INFO] loading face detector...")
detector = cv2.CascadeClassifier(args["cascade"])

image =cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("[INFO] performing face detection...")
rects = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
print("[INFO] {} faces detected...".format(len(rects)))

for (x,y,w,h) in rects:
    cv2.rectangle(image, (x,y), (x+w, y+h),(0,255,0),2)

cv2.imshow("Image", image)
cv2.waitKey(0)
