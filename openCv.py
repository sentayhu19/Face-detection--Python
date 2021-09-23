import cv2 as cv
img = cv.imread('sky.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Show face in Gray',gray)
haar_cascade =cv.CascadeClassifier('haar_face.xml')
face_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)
print(f'Number of face found ={len(face_rect)}')
for (x,y,w,h) in face_rect: 
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
cv.imshow('Detected Faces',img)
cv.waitKey(0)

#capture = cv.VideoCapture('fireworks.mp4')
#while True: 
   # isTrue,frame=capture.read()
   # cv.imshow('showing vdieo',frame)
  #  if cv.waitKey(30) & 0xFF == ord('q'):
 #    break
#capture.release()
#cv.destroyAllWindows()
