import numpy as np
import cv2 as cv
haar_cascade =cv.CascadeClassifier('haar_face.xml')
people = ['Angelina Jolie','Keanu Reeves','Charlize Theron','teddy afro']
#features = np.load('features.npy')
#lables = np.load('lables.npy')
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')
img = cv.imread(r'C:\Users\SVB\Desktop\openCv\Charlize Theron\valsvb\cvalsvb.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('person',gray)
faces_rect= haar_cascade.detectMultiScale(gray,1.1,4)
for (x,y,w,h) in  faces_rect:
    faces_rol = gray[y:y+h,x:x+h]
    label, confidence =face_recognizer.predict(faces_rol)
    print(f'lable = {people[label]} with a confidence of {confidence}')

    cv.putText(img,str(people[label]), (20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0), thickness=2)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
    cv.imshow('Detected face ',img)
    cv.waitKey(0)