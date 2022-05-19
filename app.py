from pickletools import optimize
import streamlit as st
import mysql.connector
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from streamlit_option_menu import option_menu 
import av
import cv2
import tensorflow as tf
import numpy as np
import imutils
import pickle
import os
import base64
from PIL import Image
from io import BytesIO


# faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faceCascade = cv2.CascadeClassifier("cascade2.xml")

## Video Processing function
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.i = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5)
        i =self.i+1
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (95, 207, 30), 3)
            cv2.rectangle(img, (x, y - 40), (x + w, y), (95, 207, 30), -1)
            cv2.putText(img, 'F-' + str(i), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        return img
## Generate download link
def get_image_download_link(img,filename,text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href =  f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">{text}</a>'
    return href

## Detect face
def face_detect(image,sf,mn):
    i = 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,sf,mn)
    for (x, y, w, h) in faces:
        i = i+1
        cv2.rectangle(image, (x, y), (x + w, y + h), (237, 30, 72), 3)
        cv2.rectangle(image, (x, y - 40), (x + w, y),(237, 30, 72) , -1)
        cv2.putText(image, 'F-'+str(i), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    resi_image = cv2.resize(image, (20, 20))
    return resi_image,i,image

model_path='liveness.model'
le_path='label_encoder.pickle'
encodings='encoded_faces.pickle'
detector_folder='face_detector'
confidence=0.5
args = {'model':model_path, 'le':le_path, 'detector':detector_folder, 
	'encodings':encodings, 'confidence':confidence}

# load the encoded faces and names
print('[INFO] loading encodings...')
with open(args['encodings'], 'rb') as file:
	encoded_data = pickle.loads(file.read())

# load our serialized face detector from disk
print('[INFO] loading face detector...')
proto_path = os.path.sep.join([args['detector'], 'deploy.prototxt'])
model_path = os.path.sep.join([args['detector'], 'res10_300x300_ssd_iter_140000.caffemodel'])
detector_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
	
# load the liveness detector model and label encoder from disk
liveness_model = tf.keras.models.load_model(args['model'])
le = pickle.loads(open(args['le'], 'rb').read())


class VideoProcessor:		
	def recv(self, frame):
		frm = frame.to_ndarray(format="bgr24")

		# iterate over the frames from the video stream
		# while True:
			# grab the frame from the threaded video stream
			# and resize it to have a maximum width of 600 pixels
		frm = imutils.resize(frm, width=800)

		# grab the frame dimensions and convert it to a blob
		# blob is used to preprocess image to be easy to read for NN
		# basically, it does mean subtraction and scaling
		# (104.0, 177.0, 123.0) is the mean of image in FaceNet
		(h, w) = frm.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frm, (300,300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
		
		# pass the blob through the network 
		# and obtain the detections and predictions
		detector_net.setInput(blob)
		detections = detector_net.forward()
		
		# iterate over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e. probability) associated with the prediction
			confidence = detections[0, 0, i, 2]
			
			# filter out weak detections
			if confidence > args['confidence']:
				# compute the (x,y) coordinates of the bounding box
				# for the face and extract the face ROI
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype('int')
				
				# expand the bounding box a bit
				# (from experiment, the model works better this way)
				# and ensure that the bounding box does not fall outside of the frame
				startX = max(0, startX-20)
				startY = max(0, startY-20)
				endX = min(w, endX+20)
				endY = min(h, endY+20)
				
				# extract the face ROI and then preprocess it
				# in the same manner as our training data

				face = frm[startY:endY, startX:endX] # for liveness detection
				# expand the bounding box so that the model can recog easier

				# some error occur here if my face is out of frame and comeback in the frame
				try:
					face = cv2.resize(face, (32,32)) # our liveness model expect 32x32 input
				except:
					break

				# initialize the default name if it doesn't found a face for detected faces
				name = 'Unknown'
				face = face.astype('float') / 255.0 
				face = tf.keras.preprocessing.image.img_to_array(face)

				# tf model require batch of data to feed in
				# so if we need only one image at a time, we have to add one more dimension
				# in this case it's the same with [face]
				face = np.expand_dims(face, axis=0)
			
				# pass the face ROI through the trained liveness detection model
				# to determine if the face is 'real' or 'fake'
				# predict return 2 value for each example (because in the model we have 2 output classes)
				# the first value stores the prob of being real, the second value stores the prob of being fake
				# so argmax will pick the one with highest prob
				# we care only first output (since we have only 1 input)
				preds = liveness_model.predict(face)[0]
				j = np.argmax(preds)
				label_name = le.classes_[j] # get label of predicted class
				
				# draw the label and bounding box on the frame
				label = f'{label_name}: {preds[j]:.4f}'
				print(f'[INFO] {name}, {label_name}')
				
				if label_name == 'fake':
					cv2.putText(frm, "Fake Alert!", (startX, endY + 25), 
								cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 2)
				
				cv2.putText(frm, name, (startX, startY - 35), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,130,255),2 )
				cv2.putText(frm, label, (startX, startY - 10),
							cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
				cv2.rectangle(frm, (startX, startY), (endX, endY), (0, 0, 255), 4)

		return av.VideoFrame.from_ndarray(frm, format='bgr24')


# Mysql Database CRUD Starts
# face_api_db = mysql.connector.connect(host="localhost", user="root", passwd="", database="face_api_db")
# mycursor = face_api_db.cursor()
# mycursor.execute("show databases")
# for i in mycursor:
# 	print(i)
# Mysql Database CRUD Ends


with st.sidebar:
	choice = option_menu(
		menu_title = "Homepage",
		options = ["Face Detection test", "Face Verification","Face Comparison", "Document Verification"],
	)	
def run():
	if choice == 'Face Detection test':
			img_test = st.button("Image Upload Face Detection") 
			vid_test = st.button("Live Video Face Detection")

			st.markdown(
				'''<h4 style='text-align: left; color: #d73b5c;'> Face Detection Using Image Upload"</h4>''',
					unsafe_allow_html=True)
			img_file = st.file_uploader("Choose an Image of Your ID Card in jpeg, jpg, png, jfif, or png", type=['jpg', 'jpeg', 'jfif', 'png'])
			if img_file is not None:
				img = np.array(Image.open(img_file))
				img1 = cv2.resize(img, (350, 350))
				place_h = st.columns(2)
				place_h[0].image(img1)
				st.markdown(
					'''<h4 style='text-align: left; color: #d73b5c;'>* Increase & Decrease it to get better accuracy.</h4>''',
					unsafe_allow_html=True)
				scale_factor = st.slider("Set Scale Factor Value", min_value=1.0, max_value=2.0, step=0.10, value=1.10)
				min_Neighbors = st.slider("Set Scale Min Neighbors", min_value=1, max_value=20, step=1, value=1)
				fd, count, orignal_image = face_detect(img, scale_factor, min_Neighbors)
				place_h[1].image(fd)
				if count == 0:
					st.error("No Face found!!")
				else:
					st.success("Face Found ")
					result = Image.fromarray(orignal_image)
					st.markdown(get_image_download_link(result, img_file.name, 'Download Image'), unsafe_allow_html=True)	
			st.markdown(
				'''<h4 style='text-align: left; color: #d73b5c;'>Face Detection with Camera:"</h4>''',
				unsafe_allow_html=True)
			st.markdown(
				'''<h5s style='text-align: left; color: #d73b5c;'>Instructions"</h5>''',
				unsafe_allow_html=True)
			st.markdown(
				'''<h6 style='text-align: left; color: #d73b5c;'> 1.  Do not Wear Glasses.</h6>''',
				unsafe_allow_html=True)
			st.markdown(
				'''<h6 style='text-align: left; color: #d73b5c;'> 2.  Find a Place With More light </h6>''',
				unsafe_allow_html=True)
			st.markdown(
				'''<h6 style='text-align: left; color: #d73b5c;'> 3.  Dont Shake Your Device Too Much </h6>''',
				unsafe_allow_html=True)
			webrtc_streamer(key="key", video_processor_factory=VideoProcessor,rtc_configuration={
					"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
				},sendback_audio=False, video_receiver_size=1)
	if choice == 'Face Verification':
			ver_ind = st.button("Individual Face Verification") 
			ver_db = st.button("Database Linked Face Verification")
			st.markdown(
				'''<h4 style='text-align: left; color: #d73b5c;'>* RAJ Face Detection is carried out using Haar Cascade & OpenCV"</h4>''',
				unsafe_allow_html=True)
			img_file = st.file_uploader("Choose an Image", type=['jpg', 'jpeg', 'jfif', 'png'])
			if img_file is not None:
				img = np.array(Image.open(img_file))
				img1 = cv2.resize(img, (350, 350))
				place_h = st.columns(2)
				place_h[0].image(img1)
				st.markdown(
					'''<h4 style='text-align: left; color: #d73b5c;'>* Increase & Decrease it to get better accuracy.</h4>''',
					unsafe_allow_html=True)
				scale_factor = st.slider("Set Scale Factor Value", min_value=1.0, max_value=2.0, step=0.10, value=1.10)
				min_Neighbors = st.slider("Set Scale Min Neighbors", min_value=1, max_value=20, step=1, value=1)
				fd, count, orignal_image = face_detect(img, scale_factor, min_Neighbors)
				place_h[1].image(fd)
				if count == 0:
					st.error("No Face found!!")
				else:
					st.success("Face Found ")
					result = Image.fromarray(orignal_image)
					st.markdown(get_image_download_link(result, img_file.name, 'Download Image'), unsafe_allow_html=True)
	if choice == 'Face Comparison':
			img_img =st.button("Image vs Image") 
			img_id =st.button("Image vs IDentity Card")
			vid_id= st.button("Video vs Identity Card")
			st.markdown(
				'''<h4 style='text-align: left; color: #d73b5c;'>* RAJ Face Detection is carried out using Haar Cascade & OpenCV"</h4>''',
				unsafe_allow_html=True)
			img_file = st.file_uploader("Choose an Image", type=['jpg', 'jpeg', 'jfif', 'png'])	
	if choice == 'Document Verification':
			indiv_id_read =st.button("Individual Id Verification")
			db_id_read = st.button("Database Linked IDverification")
			if indiv_id_read:
				st.markdown(
					'''<h4 style='text-align: left; color: #d73b5c;'>* Fill up the Form Then Upload Your ID"</h4>''',
					unsafe_allow_html=True)
				fname= st.text_input("First Name (required)")
				if not fname:
					st.warning("Please fill out First Name")
				midname= st.text_input("Mid name?")
				lname= st.text_input("Last Name(required)")
				if not lname:
					st.warning("Please fill out Last Name")
				st.text_input("Identity Card Number")
				img_file = st.file_uploader("Choose an Image of Your ID Card in jpeg, jpg, png, jfif, or png", type=['jpg', 'jpeg', 'jfif', 'png'])
				if img_file is not None:
					img = np.array(Image.open(img_file))
					img1 = cv2.resize(img, (350, 350))
					place_h = st.columns(2)
					place_h[0].image(img1)
					st.markdown(
						'''<h4 style='text-align: left; color: #d73b5c;'>* Increase & Decrease it to get better accuracy.</h4>''',
						unsafe_allow_html=True)
					scale_factor = st.slider("Set Scale Factor Value", min_value=1.0, max_value=2.0, step=0.10, value=1.10)
					min_Neighbors = st.slider("Set Scale Min Neighbors", min_value=1, max_value=20, step=1, value=1)
					fd, count, orignal_image = face_detect(img, scale_factor, min_Neighbors)
					place_h[1].image(fd)
					if count == 0:
						st.error("No Face found!!")
					else:
						st.success("Face Found ")
						result = Image.fromarray(orignal_image)
						st.markdown(get_image_download_link(result, img_file.name, 'Download Image'), unsafe_allow_html=True)
run()

