import cv2
from random import randrange

trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#choose img to detect face
#img=cv2.imread('A.jpg')

#to capture video from wevcam
webcam=cv2.VideoCapture(0)

while True:

	###Read the current Frame
	succeccful_frame_read,frame=webcam.read()

	###Must convert to greyscale
	greyscaled_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	##detect faces
	face_coordinates=trained_face_data.detectMultiScale(greyscaled_img)

	#Draw rectangle around the face
	for (x,y,w,h) in face_coordinates:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),3)


	cv2.imshow('Programmer Face Detection',frame)
	cv2.waitKey(1)

	
## release the VideoCapture Object
webcam.release()

print("code completed");


