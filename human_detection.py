import cv2
import imutils

#Histogram of Oriented Gradients (HOG) detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#capturing the video 
cap = cv2.VideoCapture('humandet_video.mp4')

#opening the file 
while cap.isOpened():
#begins the loop over the frames 
    ret, image = cap.read()
    if ret:
        image = imutils.resize(image,
                               #to check if the frame is read successfully 
                            width=min(400, image.shape[1]))

         #resizing the frame width 
        (regions, _) = hog.detectMultiScale(image, winStride=(4, 4), padding=(4, 4), scale=1.05)
#human detection starts 

        for (x, y, w, h) in regions:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Showing the output Image
        cv2.imshow("Image", image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()





