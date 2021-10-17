import os
import cv2 as cv
import numpy as np
people = ['Angelina Jolie','Keanu Reeves','Charlize Theron','teddy afro']
print(people)
DIR =r'C:\Users\SVB\Desktop\openCv'
haar_cascade =cv.CascadeClassifier('haar_face.xml')
features=[]
labeles=[]
def  create_train():
    for person in people:
        path=os.path.join(DIR,person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,
            minNeighbors=4)
            for (x,y,w,h ) in faces_rect:
                faces_roi = gray[y:y+h,x:x+w]
                features.append(faces_roi)
                labeles.append(label)
create_train()
print("*** Trianing AI Done ***")
features = np.array(features,dtype='object')
labeles =np.array(labeles)
#print(f'length of the features = {len(features)}')
#print(f'length of the labels = {len(labeles)}')
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features,labeles)
face_recognizer.save('face_trained.yml')
np.save('features.npy',features)
np.save('lables.npy',labeles)