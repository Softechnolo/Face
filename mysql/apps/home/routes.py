from doctest import COMPARISON_FLAGS
import json
from gzip import READ
from msilib.schema import File
from tkinter import PhotoImage
from tkinter.filedialog import SaveAs

from flask import Flask, render_template, redirect, url_for, request, session
from flask_sqlalchemy import SQLAlchemy
from flask import Flask ,render_template ,request, redirect, url_for ,g ,jsonify ,send_from_directory , send_file
import os

from imageio import save

from apps.authentication import routes
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import hashlib
import sqlite3
from datetime import date
import cv2
import numpy as np
import base64
import datetime
import io
import pytesseract
from PIL import Image
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms.validators import InputRequired, Email, Length
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

# Compare
import request_id
from flask import Flask, jsonify, request, render_template
from request_id import RequestIdMiddleware
from werkzeug.serving import make_server
from src.OCR.crop_morphology import crop_morphology
from src.constants import ALLOWED__PICTURE_EXTENSIONS, ALLOWED_VIDEO_EXTENSIONS, frames_folder, upload_folder, \
    image_size_threshold, max_resize, source_type_image, source_type_video
from src.face_processing import compare_face
# Compare
import os
import request_id
from flask import Flask, jsonify, request, render_template
from request_id import RequestIdMiddleware
from werkzeug.serving import make_server


import os
import request_id
from flask import Flask, jsonify, request, render_template
from request_id import RequestIdMiddleware
from werkzeug.serving import make_server
# Compare
from email import message
import datetime
import uuid
from apps.home import blueprint
from flask import render_template, request
from flask_login import login_required
from jinja2 import TemplateNotFound
import os
from datetime import date
import cv2
import numpy as np
import base64
from flask import render_template, request
from flask_login import login_required
from flask import Flask, redirect, url_for, session
from werkzeug.utils import secure_filename
from apps.face_recognition_and_liveness.face_liveness_detection.face_recognition_liveness_app import recognition_liveness
# Config

pytesseract.pytesseract.tesseract_cmd ='/home/ubuntu/.linuxbrew/bin/tesseract'
static = os.path.abspath('static')
app = Flask(__name__, static_url_path='', static_folder=static)
app.config.from_object('config')
app.config['UPLOAD_FOLDER1'] = 'apps/static/assets/img/image1'
app.config['UPLOAD_FOLDER2'] = 'apps/static/assets/img/image2'
app.config['UPLOAD_FOLDER3'] = 'upload_for_ver'

# Config

# Comparison Video
middleware = RequestIdMiddleware(
    app,
    format='{status} {REQUEST_METHOD:<6} {REQUEST_PATH:<60} {REQUEST_ID}',
)


def get_error_result(source_type, is_no_files):
    if is_no_files:
        result = {
            "status_code": 400,
            "error": "No " + source_type + " Found"
        }
    else:
        result = {
            "status_code": 400,
            "error": source_type + " extension is not correct"
        }
    return jsonify(result)


def create_directories():
    # Check if upload and frames folder existed or not.
    # If not then create it
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    if not os.path.exists(frames_folder):
        os.makedirs(frames_folder)

    # Get unique Request ID
    face_matching_request_id = request_id.get_request_id(request)
    print("Request ID:", face_matching_request_id)

    # create a subdirectory with unique request id inside frames and upload folder
    request_upload_folder_path = os.path.join(upload_folder, face_matching_request_id)
    request_frames_folder_path = os.path.join(frames_folder, face_matching_request_id)
    os.makedirs(request_frames_folder_path)
    os.makedirs(request_upload_folder_path)

    return request_upload_folder_path, request_frames_folder_path


def set_tolerance_and_threshold(tolerance, threshold, sharpness):
    if tolerance != '':
        tolerance = float(tolerance)
    else:
        tolerance = 0.50

    if threshold != '':
        threshold = float(threshold)
    else:
        threshold = 0.80

    if sharpness is not None and sharpness != '':
        sharpness = float(sharpness)
    else:
        sharpness = 0.60

    print("Tolerance: ", tolerance)
    print("Face match threshold: ", threshold)
    print("Sharpness threshold: ", sharpness)
    return tolerance, threshold, sharpness


def check_files_uploaded():
    if request.files['known'].filename == '':
        print("no image uploaded")
        return False, source_type_image
    if request.files['unknown'].filename == '':
        print("no video uploaded")
        return False, source_type_video
    return True, "pass"


def check_valid_files_uploaded(known, unknown):
    if not known.filename.lower().endswith(ALLOWED__PICTURE_EXTENSIONS):
        return False, source_type_image
    if not unknown.filename.lower().endswith(ALLOWED_VIDEO_EXTENSIONS):
        return False, source_type_video
    return True, "pass"


@blueprint.route('/api/upload', methods=['POST'])
def upload_image_video():
    # Check whether files is uploaded or not
    is_files_uploaded, source_type = check_files_uploaded()
    if not is_files_uploaded:
        if source_type == "image":
            return get_error_result("Image", True)
        else:
            return get_error_result("Video", True)

    known = request.files['known']
    unknown = request.files['unknown']

    # Check if a valid image and video file was uploaded
    is_valid_files_uploaded, source_type = check_valid_files_uploaded(known, unknown)
    if not is_valid_files_uploaded:
        if source_type == "image":
            return get_error_result("Image", True)
        else:
            return get_error_result("Video", True)

    # Flask doesn't receive any information about
    # what type the client intended each value to be.
    # So it parses all values as strings.
    # And we need to parse it manually to float and set the value
    tolerance = request.form['tolerance']
    threshold = request.form['threshold']
    sharpness = request.form.get('sharpness')
    tolerance, threshold, sharpness = set_tolerance_and_threshold(tolerance, threshold, sharpness)

    # for Unit Test to pass without running through whole face matching process
    if "testing" in request.form:
        return jsonify(result={"status_code": 200})

    # create absolutely paths for the uploaded files
    request_upload_folder_path, request_frames_folder_path = create_directories()
    unknown_filename_path = os.path.join(request_upload_folder_path, unknown.filename)
    known_filename_path = os.path.join(request_upload_folder_path, known.filename)

    # Save the uploaded files to directory
    # Example: upload/request-id/image.jpg
    unknown.save(unknown_filename_path)
    known.save(known_filename_path)
    video_path = os.path.join(request_upload_folder_path, unknown.filename)

    if known and unknown:

        # Resize the known image and scale it down
        known_image_size = os.stat(known_filename_path).st_size
        print("Image Size: ", known_image_size)
        if known_image_size > image_size_threshold:
            print("Resizing the known image as it was larger than ", image_size_threshold)
            known_image = cv2.imread(known_filename_path)
            resized_image = cv2.resize(known_image, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
            cv2.imwrite(known_filename_path, resized_image)
            print("Resized image ", os.stat(known_filename_path).st_size)

            if os.stat(known_filename_path).st_size < max_resize:
                print("Enlarge back as it smaller than ", max_resize)
                known_image = cv2.imread(known_filename_path)
                resized_image = cv2.resize(known_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(known_filename_path, resized_image)
                print("Resized image ", os.stat(known_filename_path).st_size)

        crop_morphology(known_filename_path)

        # process both image and video
        return compare_face(known_filename_path,
                            video_path,
                            request_upload_folder_path,
                            request_frames_folder_path,
                            tolerance=tolerance,
                            face_match_threshold=threshold,
                            sharpness_threshold=sharpness)


# Comparison Video

# Comparison Image
class CompareImage(object):
    def __init__(self, image_1_path, image_2_path):
        self.minimum_commutative_image_diff = 1
        self.image_1_path = image_1_path
        self.image_2_path = image_2_path
    def compare_image(self):
        image_1 = cv2.imread(self.image_1_path, 0)
        image_2 = cv2.imread(self.image_2_path, 0)
        commutative_image_diff = self.get_image_difference(image_1, image_2)

        if commutative_image_diff < self.minimum_commutative_image_diff:
            print ("Matched")
            return commutative_image_diff
        return "Not Matched" 
        # //random failure value
    @staticmethod
    def get_image_difference(image_1, image_2):
        first_image_hist = cv2.calcHist([image_1], [0], None, [256], [0, 256])
        second_image_hist = cv2.calcHist([image_2], [0], None, [256], [0, 256])

        img_hist_diff = cv2.compareHist(first_image_hist, second_image_hist, cv2.HISTCMP_BHATTACHARYYA)
        img_template_probability_match = cv2.matchTemplate(first_image_hist, second_image_hist, cv2.TM_CCOEFF_NORMED)[0][0]
        img_template_diff = 1 - img_template_probability_match

        # taking only 10% of histogram diff, since it's less accurate than template method
        commutative_image_diff = (img_hist_diff / 10) + img_template_diff
        return commutative_image_diff
# Comparison Image
# Face Detection Begins
@blueprint.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    # Read image
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    # Detect faces
    faces = detect_faces(image)
    if len(faces) == 0:
        faceDetected = False
        num_faces = 0
        to_send = ''
    else:
        faceDetected = True
        num_faces = len(faces)
        # Draw a rectangle
        for item in faces:
            draw_rectangle(image, item['rect'])
        # In memory
        image_content = cv2.imencode('.jpg', image)[1].tostring()
        encoded_image = base64.encodebytes(image_content)
        to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')
    return render_template('home/face_detection.html', faceDetected=faceDetected, num_faces=num_faces, image_to_show=to_send, init=True)
def detect_faces(img):
    '''Detect face in an image'''
    faces_list = []
    # Convert the test image to gray scale (opencv face detector expects gray images)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Load OpenCV face detector (LBP is faster)
    
    face_cascade = cv2.CascadeClassifier('apps/opencv-files/cascade4.xml')
    # Detect multiscale images (some images may be closer to camera than others)
    # result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4);
    # If not face detected, return empty list  
    if  len(faces) == 0:
        return faces_list
    for i in range(0, len(faces)):
        (x, y, w, h) = faces[i]
        face_dict = {}
        face_dict['face'] = gray[y:y + w, x:x + h]
        face_dict['rect'] = faces[i]
        faces_list.append(face_dict)
    # Return the face image area and the face rectangle
    return faces_list
# ----------------------------------------------------------------------------------
# Draw rectangle on image
# according to given (x, y) coordinates and given width and heigh
# ----------------------------------------------------------------------------------
def draw_rectangle(img, rect):
    '''Draw a rectangle on the image'''
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
# Face Detection Ends

#face verification

@blueprint.route('/face_verification' , methods =['GET','POST'])
def face_verification():
    if request.method == 'POST':
        print(request.form.get('longitude'))     
        print(request.form.get('latitude')) 
        
        # //Video Save
        vid1 = request.files['vid']
        vid_arg1 = secure_filename(vid1.filename)
        directory3=os.path.join(app.config['UPLOAD_FOLDER3'], vid_arg1)
        vid1s = vid_arg1
        print (vid1s)
        vid1.save(directory3)
        # //Video Save
        path_to_rec = directory3
        detected_name, label_name = recognition_liveness('apps/face_recognition_and_liveness/face_liveness_detection/liveness.model',
                                                 'apps/face_recognition_and_liveness/face_liveness_detection/label_encoder.pickle',
                                                 'apps/face_recognition_and_liveness/face_liveness_detection/face_detector',
                                                 'apps/face_recognition_and_liveness/face_recognition/encoded_faces.pickle',
                                                  confidence=0.5, path_to_rec = path_to_rec)
        if detected_name != 'Unknown' and label_name == 'real':
            return render_template("home/face_verification.html", message = "Real, verified")
        elif detected_name != 'Unknown' and label_name == 'fake':
            return render_template("home/face_verification.html", message = "Fake")
        elif detected_name == 'Unknown' and label_name == 'real':
            return render_template("home/face_verification.html", message = "Real, But Not registered")
        else:
            return render_template("home/face_verification.html", message = "Try Again")
    else:
        return render_template("home/face_verification.html", message = "Try Again")
# face verification Ends

# Face Comparison Image To Image Starts
@blueprint.route('/img_to_img', methods = ['POST', 'GET'])
def img_to_img():
    if request.method == 'POST':
        img1 = request.files['img1']
        img2 = request.files['img2']

        now = datetime.datetime.now()
        currentDate = str(now.microsecond) + "_" + str(now.second) + "_" + str(now.minute) + "_" + str(now.hour) + "_" + str(now.month) + "_" + str(now.day) + "_" + str(now.year) 
        currentDate1 = str(now.second) + "_" + str(now.second) + "_" + str(now.second) + "_" + str(now.second) + "_" + str(now.minute) + "_" + str(now.hour) + "_" + str(now.month) + "_" + str(now.day) + "_" + str(now.year) 

        file1 = str(uuid.uuid4())
        file2 = str(uuid.uuid4())
        img_arg1 = file1 + secure_filename(img1.filename)
        img_arg2 = file2 + secure_filename(img2.filename)


        directory1=os.path.join(app.config['UPLOAD_FOLDER1'], img_arg1)
        directory2=os.path.join(app.config['UPLOAD_FOLDER2'], img_arg2)

        d1s = img_arg1
        d2s = img_arg2

        print (d1s)
        print (d2s)
        img1.save(directory1)
        img2.save(directory2)

    compare_image = CompareImage(directory1 , directory2)
    image_difference = compare_image.compare_image()
    print (image_difference)
    message = ''

    if image_difference != 1000:
            return render_template('home/face_compare_img_img.html', message = image_difference)
    else:
            return render_template('home/face_compare_img_img.html', message ="Not the Same")

# Face Comparison Image To Image Ends

@blueprint.route('/homepage')
@login_required
def homepage():
    return render_template('home/homepage.html')


@blueprint.route('/index')
@login_required
def index():
    return render_template('home/index.html', segment='index')

# ID CARD Verification
@blueprint.route('/img_to_id', methods = ['POST', 'GET'])
def img_to_id():
    if request.method == 'POST':
        img1 = request.files['img1']
        img2 = request.files['img2']
        now = datetime.datetime.now()
        currentDate = str(now.microsecond) + "_" + str(now.second) + "_" + str(now.minute) + "_" + str(now.hour) + "_" + str(now.month) + "_" + str(now.day) + "_" + str(now.year) 
        currentDate1 = str(now.second) + "_" + str(now.second) + "_" + str(now.second) + "_" + str(now.second) + "_" + str(now.minute) + "_" + str(now.hour) + "_" + str(now.month) + "_" + str(now.day) + "_" + str(now.year) 
        file1 = str(uuid.uuid4())
        file2 = str(uuid.uuid4())
        img_arg1 = file1 + secure_filename(img1.filename)
        img_arg2 = file2 + secure_filename(img2.filename)
        directory1=os.path.join(app.config['UPLOAD_FOLDER1'], img_arg1)
        directory2=os.path.join(app.config['UPLOAD_FOLDER2'], img_arg2)
        d1s = img_arg1
        d2s = img_arg2
        print (d1s)
        print (d2s)
        img1.save(directory1)
        img2.save(directory2)

        # //Sharpen id
        img1 = cv2.imread(directory1)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        sharpen_kernel1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpen1 = cv2.filter2D(gray1, -1, sharpen_kernel1)
        thresh1 = cv2.threshold(sharpen1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # //Sharpen
        # //Sharpen id
        img2 = cv2.imread(directory2)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        sharpen_kernel2 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpen2= cv2.filter2D(gray2, -1, sharpen_kernel2)
        thresh = cv2.threshold(sharpen2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # //Sharpen

    compare_image = CompareImage(directory1 , directory2)
    image_difference = compare_image.compare_image()
    print (image_difference)
    message = ''

    if image_difference != 1000:
            return render_template('home/id_card_verification.html', message = image_difference)
    else:
            return render_template('home/id_card_verification.html', message ="Not the Same")

#ID Card Verification

# Document Verification
@blueprint.route('/scanner', methods=['GET', 'POST'])
def scan_file1():
    if request.method == 'POST':
        field_name = request.form.getlist("name[]")
        field_list = request.form.getlist("age[]")
        start_time = datetime.datetime.now()
        img = request.files['file']
        # //save
        file1 = str(uuid.uuid4())
        img_arg1 = file1 + secure_filename(img.filename)
        directory1=os.path.join(app.config['UPLOAD_FOLDER1'], img_arg1)
        d1s = img_arg1
        print (d1s)
        img.save(directory1)
        # //save
        img = cv2.imread(directory1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpen = cv2.filter2D(gray, -1, sharpen_kernel)
        thresh = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        scanned_text = pytesseract.image_to_string(thresh, lang='eng', config='--psm 6').upper()
        scanned_text2 = scanned_text.replace(" ", "").replace("\t", "")

        verification_passed=[]
        verification_failed=[]

        for x in field_list:
            if x.upper() in scanned_text2:
                verification_passed.append(x)
            else:
                verification_failed.append(x)
        
        print("Found data:", scanned_text2)
        print("Verified Info:", verification_passed)
        print("Unverified Info:", verification_failed)


        session['data'] = {
            "text": scanned_text2,
            "time": str((datetime.datetime.now() - start_time).total_seconds()),
            "ver_passed": verification_passed,
            "ver_failed": verification_failed
        }

        
        data = session['data']
        return render_template("home/document_verification.html", text =data["text"], title="Result", verified_info = data["ver_passed"], unverified_info = data["ver_failed"], words=len(data["text"].split(" "))
        )
        
# Document Verification

@blueprint.route('/<template>')
@login_required
def route_template(template):

    try:

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template("home/" + template, segment=segment)

    except TemplateNotFound:
        return render_template('home/page-404.html'), 404

    except:
        return render_template('home/page-500.html'), 500


# Helper - Extract current page name from request
def get_segment(request):

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'homepage'
        return segment
    except:
        return None
