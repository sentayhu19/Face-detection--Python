import cv2 as cv
img = cv.imread('ja.webp')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Show face',gray)
haar_cascade =cv.CascadeClassifier('haar_face.xml')
face_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)
print(f'Number of face found ={len(face_rect)}')
cv.waitKey(0)
#capture = cv.VideoCapture('fireworks.mp4')
#while True:
   # isTrue,frame=capture.read()
   # cv.imshow('showing vdieo',frame)
  #  if cv.waitKey(30) & 0xFF == ord('q'):
 #    break
#capture.release()
#cv.destroyAllWindows()
