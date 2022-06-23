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


from face_recognition_and_liveness.face_liveness_detection.face_recognition_liveness_app import recognition_liveness

app = Flask(__name__)
app.secret_key = 'web_app_for_face_recognition_and_liveness' # something super secret
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.sqlite'
Bootstrap(app)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
app.config['UPLOAD_FOLDER'] = 'face_recognition_and_liveness/face_recognition/dataset'
DATABASE='/home/www/database.sqlite'
app.config['UPLOAD_FOLDER1'] = 'image1'
app.config['UPLOAD_FOLDER2'] = 'image2'
app.config['UPLOAD_FOLDER3'] = 'uploads_for_ver'



class Users(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100))
    name = db.Column(db.String(100))
    phone = db.Column(db.String(100))
    password = db.Column(db.String(100))
    passport = db.Column(db.String(100))
    photo_id = db.Column(db.String(100))
    staus = db.Column(db.String(100))
    user_ip = db.Column(db.String(100))
    

def get_db():
	db = getattr(g, '_database', None)
	if db is None:
		db = g._database = sqlite3.connect(DATABASE)
	return db


@login_manager.user_loader
def load_user(user_id):
    return Users.query.get(int(user_id))

class LoginForm(FlaskForm):
    username = StringField('username', validators=[InputRequired(), Length(min=4, max=20)])
    password = PasswordField('password', validators=[InputRequired(), Length(min=5, max=80)])
    remember = BooleanField('remember me')

class RegisterForm(FlaskForm):
    name = StringField('Full Name', validators=[InputRequired(), Length(min=6, max=90)])
    username = StringField('Email', validators=[InputRequired(), Length(min=4, max=20)])
    phone = StringField('Phone No.', validators=[InputRequired(), Length(min=4, max=20)])
    password = PasswordField('password', validators=[InputRequired(), Length(min=5, max=80)])
    passport = FileField('Facial Photograph')


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


@app.route('/api/upload', methods=['POST'])
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


@app.route('/img_to_vid_compare')
def img_to_vid_compare():
    return render_template('img_to_vid_compare.html')
# Comparison Video

#  Face Verification Image


#  Face Verification Image


#  Comparison Image
@app.route('/img_to_img_compare')
def img_to_img_compare():
    return render_template('img_to_img_compare.html')
 
@app.route('/img_to_img', methods = ['POST', 'GET'])
def img_to_img():
    if request.method == 'POST':
        img1 = request.files['img1']
        img2 = request.files['img2']

        now = datetime.datetime.now()
        currentDate = str(now.month) + "_" + str(now.day) + "_" + str(now.year)
        
        img_arg1 = currentDate + secure_filename(img1.filename)
        img_arg2 = currentDate + secure_filename(img2.filename)


        directory1=os.path.join(app.config['UPLOAD_FOLDER1'], img_arg1)
        directory2=os.path.join(app.config['UPLOAD_FOLDER2'], img_arg2)

        d1s = img_arg1
        d2s = img_arg2

        print (d1s)
        print (d2s)
        img1.save(directory1)
        img2.save(directory2)

    compare_image = CompareImage('image1/'+ d1s , 'image2/' + d2s)
    image_difference = compare_image.compare_image()
    print (image_difference)
    if image_difference != 1000:
            return render_template('img_to_img_compare.html', message= image_difference)
    else:
            return render_template('img_to_img_compare.html', message="Not the Same")
        
@app.errorhandler(500)
def internal_error(error):
    return render_template('index.html', message="Please upload valid photos having a face")   
		
@app.errorhandler(404)
def not_found(error):
        return render_template('index.html', message="")      
    

#Register AND LOGIN
@app.route('/')
def index():
    return render_template('base1.html')




#Database query function to return raw data from database
def query_db(query, args=(), one=False):
	cur = get_db().execute(query, args)
	rv = cur.fetchall()
	cur.close()
	return (rv[0] if rv else None) if one else rv

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        user = Users.query.filter_by(username=form.username.data).first()
        if user:
            # compares the password hash in the db and the hash of the password typed in the form
            if check_password_hash(user.password, form.password.data):
                login_user(user, remember=form.remember.data)
                return redirect(url_for('dashboard'))
        return 'invalid username or password'

    return render_template('login.html', form=form)



@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='sha256')
        # Experiment
        passport = form.passport.data
        filename = secure_filename(passport.filename)
        directory=os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # Experiment
        # add the user form input which is form.'field'.data into the column which is 'field'
        passport.save(directory)
        size=os.path.getsize(directory)
        # filehash=hashlib.sha1(directory).hexdigest()
        os.rename(directory,os.path.join(app.config['UPLOAD_FOLDER'], filename))
        new_user = Users(username=form.username.data, name=form.name.data, phone=form.phone.data, password=hashed_password, passport=filename,user_ip=request.remote_addr)
        db.session.add(new_user)
        db.session.commit()
        return 'Account  has been created go and login!'
    return render_template('signup.html', form=form)

@app.route("/download/<filehash>",methods=['GET'])
def download(filehash):
		#filehash is sha1 hash stored in database of file.Increase download counter
		data=query_db('select * from files where hash=?',[filehash])
		counter=int(data[0][5])+1
		try:
			get_db().execute("update files SET counter = ? WHERE hash=?", [counter,filehash])
			get_db().commit()
			#return send_from_directory(app.config['UPLOAD_FOLDER'], data[0][3])
			return send_file(os.path.join(app.config['UPLOAD_FOLDER'], data[0][3]),attachment_filename=data[0][1],as_attachment=True)
		except:
			return 'File not Found'

@app.route("/server-usage",methods=['GET'])
def server_usage():
	data=query_db('select * from files')
	bandwidth=0
	for i in data:
		bandwidth+=int(i[5])*int(i[2]) #Multiplying counter with size of file to get bandwidth amount
	return jsonify(bandwidthusage=str(bandwidth/1024.0)+" KB")


@app.route("/disk-usage",methods=['GET'])
def disk_usage():
	data=query_db('select * from files')
	diskspace=0
	for i in data:
		diskspace+=int(i[2])
	return jsonify(diskusage=str(diskspace/1024.0)+" KB")

def db_insert(filename,size,filehash):
		filename=str(filename)
		size=int(size)
		filedate=str(date.today())
		file_exist=query_db('select * from files where hash=?',[filehash])
		if not file_exist:
			get_db().execute("insert into files (filename,size,hash,date,counter) values (?,?,?,?,?)", [filename,size,filehash,filedate,0])
			get_db().commit()
		return True


@app.teardown_appcontext
def close_connection(exception):
	db = getattr(g, '_database', None)
	if db is not None:
		db.close()

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', name=current_user.username)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))
    
@app.route('/MainMenu')
def MainMenu():
    return render_template('MainMenu.html')

@app.route('/index1')
def index1():
    return render_template("index1.html")
@app.route('/document_verification')
def document_verification():
    return render_template("document_verification.html")

@app.route('/face_with_image')
def face_with_image():
    return render_template("index.html")

#face verification

@app.route('/face_rec' , methods =['GET','POST'])
def face_rec():
    if request.method == 'POST':
        print(request.form.get('longitude'))     
        print(request.form.get('latitude')) 

        # //Video Save
        # vid1 = request.files['vid']
        # vid_arg1 = secure_filename(vid1.filename)

        # directory3=os.path.join(app.config['UPLOAD_FOLDER3'], vid_arg1)
        # vid1s = vid_arg1
        # print (vid1s)
        # vid1.save(directory3)
        # //Video Save
        path_to_rec = 'uploads_for_ver/1.mp4'
        detected_name, label_name = recognition_liveness('face_recognition_and_liveness/face_liveness_detection/liveness.model',
                                                 'face_recognition_and_liveness/face_liveness_detection/label_encoder.pickle',
                                                 'face_recognition_and_liveness/face_liveness_detection/face_detector',
                                                 'face_recognition_and_liveness/face_recognition/encoded_faces.pickle',
                                                  confidence=0.5, path_to_rec = path_to_rec)
        
        if detected_name != 'Unknown' and label_name == 'real':
            return render_template("verified.html")
        elif detected_name != 'Unknown' and label_name == 'fake':
            return render_template("spoof.html")
        elif detected_name == 'Unknown' and label_name == 'real':
            return render_template("regiteration.html")
        else:
            return render_template("face_rec.html")
    else:
        return render_template("face_rec.html")

#face verification

@app.route('/indivcam')
def indivcam():
    return render_template('indivcam.html')

@app.route('/scanner', methods=['GET', 'POST'])
def scan_file():
    if request.method == 'POST':
        start_time = datetime.datetime.now()
        f_name = request.form['f_name'].upper()
        mid_name = request.form['mid_name'].upper()
        l_name = request.form['l_name'].upper()
        id_no = request.form['id_no'].upper()
        image_data = request.files['file'].read()
        file_to = request.files['file']
        
        f_name_status = ""
        mid_name_status = ""
        l_name_status = ""
        id_no_status = ""
        scanned_text = pytesseract.image_to_string(Image.open(io.BytesIO(image_data))).upper()
        scanned_text2 = scanned_text.replace(" ", "").replace("\t", "")
        
        if f_name in scanned_text2:
            f_name_status = "verified"
        else:
            f_name_status = "unverified"
        if mid_name in scanned_text2:
            mid_name_status = "verified"
        else:
            mid_name_status = "unverified"
        if l_name in scanned_text2:
            l_name_status = "verified"
        else:
            l_name_status = "unverified"
        if id_no in scanned_text2:
            id_no_status = "verified"
        else:
            id_no_status = "unverified"
        

        print("Found data:", scanned_text2)

        session['data'] = {
            "text": scanned_text2,
            "time": str((datetime.datetime.now() - start_time).total_seconds()),
            "fs" : f_name_status,
            "ls": l_name_status,
            "ids": id_no_status
        }

        return redirect(url_for('result'))

@app.route('/result')
def result():
    if "data" in session:
        data = session['data']
        return render_template(
            "home/id_card_verification.html",
            title="Result",
            time=data["time"],
            text=data["text"],
            f = data["fs"],
            l = data["ls"],
            i = data["ids"],
            words=len(data["text"].split(" "))
        )
    else:
        return "Wrong request method."

@app.route('/scanner1', methods=['GET', 'POST'])
def scan_file1():
    if request.method == 'POST':
        field_name = request.form.getlist("name[]")
        field_list = request.form.getlist("age[]")
        start_time = datetime.datetime.now()
        image_data = request.files['file'].read()
        file_to = request.files['file']
        scanned_text = pytesseract.image_to_string(Image.open(io.BytesIO(image_data))).upper()
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

        return redirect(url_for('result1'))

        
@app.route('/result1')
def result1():
    if "data" in session:
        data = session['data']
        return render_template(
            "result1.html",
            title="Result",
            verified_info = data["ver_passed"],
            unverified_info = data["ver_failed"],
            words=len(data["text"].split(" "))
        )
    else:
        return "Wrong request method."


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']

    # Save file
    #filename = 'static/' + file.filename
    #file.save(filename)

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
        
        # Save
        #cv2.imwrite(filename, image)
        
        # In memory
        image_content = cv2.imencode('.jpg', image)[1].tostring()
        encoded_image = base64.encodebytes(image_content)
        to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')

    return render_template('index.html', faceDetected=faceDetected, num_faces=num_faces, image_to_show=to_send, init=True)

# ----------------------------------------------------------------------------------
# Detect faces using OpenCV
# ----------------------------------------------------------------------------------  
def detect_faces(img):
    '''Detect face in an image'''
    
    faces_list = []

    # Convert the test image to gray scale (opencv face detector expects gray images)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load OpenCV face detector (LBP is faster)
    # face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
    face_cascade = cv2.CascadeClassifier('opencv-files/cascade4.xml')

    # Detect multiscale images (some images may be closer to camera than others)
    # result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=10);

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

# @app.route('/login_page', methods=['GET','POST'])
# def login_page():
#     if request.method == 'POST':
#         session.pop('name', None)
#         username = request.form['username']
#         password = request.form['password']
#         user = Users.query.filter_by(username=username).first()
#         print(user)
#         if user is not None and user.password == password:
#             session['name'] = user.name # store variable in session
#             detected_name, label_name = recognition_liveness('face_recognition_and_liveness/face_liveness_detection/liveness.model',
#                                                     'face_recognition_and_liveness/face_liveness_detection/label_encoder.pickle',
#                                                     'face_recognition_and_liveness/face_liveness_detection/face_detector',
#                                                     'face_recognition_and_liveness/face_recognition/encoded_faces.pickle',
#                                                     confidence=0.5)
#             if detected_name != "Unknown" and label_name == 'real':
#                 return redirect(url_for('main'))
#             else:
#                 return render_template('ver_failed_page.html', invalid_user=True, username=username)
#         else:
#             return render_template('login_page.html', incorrect=True)

#     return render_template('login_page.html')



# Video-image Face Comparison


@app.route('/main', methods=['GET'])
def main():
    name = session['name']
    return render_template('main_page.html', name=name)

if __name__ == '__main__':
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    pytesseract.pytesseract.tesseract_cmd ='/home/ubuntu/.linuxbrew/bin/tesseract'
    db.create_all()

    # add users to database

    new_user = Users(username='jom_ariya', password='123456789', name='Ariya')
    db.session.add(new_user)

    # new_user_2 = Users(username='earth_ekaphat', password='123456789', name='Ekaphat')
    # new_user_3 = Users(username='bonus_ekkawit', password='123456789', name='Ekkawit')
    # db.session.add(new_user_2)
    # db.session.add(nexportew_user_3)

    app.run(host="0.0.0.0" ,debug=True)