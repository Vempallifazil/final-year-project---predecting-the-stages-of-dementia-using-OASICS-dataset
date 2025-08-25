from flask import Flask,render_template,redirect,request,url_for, send_file, session, Response, jsonify
import mysql.connector, joblib, random, string, base64, pickle
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from keras.models import load_model

app = Flask(__name__)
app.secret_key = 'dimentia' 

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    port="3306",
    database='dimentia'
)

mycursor = mydb.cursor()

def executionquery(query,values):
    mycursor.execute(query,values)
    mydb.commit()
    return

def retrivequery1(query,values):
    mycursor.execute(query,values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']

        if password == c_password:
            query = "SELECT email FROM users"
            email_data = retrivequery2(query)
            email_data_list = []
            for i in email_data:
                email_data_list.append(i[0])

            if email not in email_data_list:
                query = "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)"
                values = (name, email, password)
                executionquery(query, values)

                return render_template('login.html', message="Successfully Registered!")
            return render_template('register.html', message="This email ID is already exists!")
        return render_template('register.html', message="Conform password is not match!")
    return render_template('register.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']

        if email.lower() == "admin@gmail.com" and password == "admin":
            return redirect("/admin")
        
        query = "SELECT email FROM users"
        email_data = retrivequery2(query)
        email_data_list = []
        for i in email_data:
            email_data_list.append(i[0])

        if email in email_data_list:
            query = "SELECT * FROM users WHERE email = %s"
            values = (email,)
            password__data = retrivequery1(query, values)
            if password == password__data[0][3]:
                session["user_email"] = email
                session["user_id"] = password__data[0][0]
                session["user_name"] = password__data[0][1]

                return redirect("/home")
            return render_template('login.html', message= "Invalid Password!!")
        return render_template('login.html', message= "This email ID does not exist!")
    return render_template('login.html')


# Patient panel

@app.route('/home')
def home():
    return render_template('patient/home.html')

@app.route('/relatives')
def relatives():
    user_id = session["user_id"]
    query = "SELECT * FROM relatives WHERE patient_id = %s"
    values = (user_id, )
    relatives_data = retrivequery1(query, values)

    relatives_list = []
    for item in relatives_data:
        relatives_list.append({
            'id': item[0],
            'name': item[1],
            'img': base64.b64encode(item[2]).decode('utf-8'),
            'relation': item[3],
            'description': item[4],
            'audio': base64.b64encode(item[5]).decode('utf-8'),
            'patient_id': item[6],
            'mobile': item[7]
        })

    return render_template('patient/relatives.html', patient_id = user_id, relatives_data = relatives_list)


@app.route('/condition')
def condition():
    user_name = session["user_name"]
    user_id = session["user_id"]
    query = "SELECT * FROM patient_condition WHERE patient_id = %s ORDER BY id DESC LIMIT 1"
    values = (user_id, )
    condition_data = retrivequery1(query, values)

    if condition_data:
        img = base64.b64encode(condition_data[0][1]).decode('utf-8')
        return render_template('patient/condition.html', data = condition_data, user_name = user_name, img = img)
    return render_template('patient/condition.html')


# Admin panel

@app.route('/admin')
def admin():
    return render_template('admin/admin.html')


@app.route('/patients')
def patients():
    query = "SELECT * FROM users ORDER BY name ASC"
    data = retrivequery2(query,)
    return render_template('admin/patients.html', data = data)



@app.route('/add_relatives/<patient_id>', methods = ["GET", "POST"])
def add_relatives(patient_id):
    if request.method == "POST":
        name = request.form["name"]
        relation = request.form["relation"]
        description = request.form["description"]
        mobile = request.form["mobile"]

        img = request.files["img"]
        binary_data = img.read()

        audio = request.files["audio"]
        audio_data = audio.read()

        query = "INSERT INTO relatives (name, img, relation, description, audio, patient_id, mobile) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        values = (name, binary_data, relation, description, audio_data, patient_id, mobile)
        executionquery(query, values)

        return render_template('admin/add_relatives.html', patient_id = patient_id, message="Information added successfully!")
    return render_template('admin/add_relatives.html', patient_id = patient_id)


@app.route('/manage_realatives/<patient_id>', methods = ["GET", "POST"])
def manage_realatives(patient_id):
    message = None
    if request.method == "POST":
        relative_id = request.form["id"]
        name = request.form["name"]
        relation = request.form["relation"]
        description = request.form["description"]
        mobile = request.form["mobile"]
        img = request.files["img"]

        if img:
            binary_data = img.read()
            query = "UPDATE relatives SET name = %s, img = %s, relation = %s, description = %s, mobile = %s WHERE id = %s"
            values = (name, binary_data, relation, description, mobile, relative_id)
        else:
            query = "UPDATE relatives SET name = %s, relation = %s, description = %s, mobile = %s WHERE id = %s"
            values = (name, relation, description, mobile, relative_id)

        executionquery(query, values)
        message = "Updated successfully!"


    query = "SELECT * FROM relatives WHERE patient_id = %s"
    values = (patient_id, )
    relatives_data = retrivequery1(query, values)

    relatives_list = []
    for item in relatives_data:
        relatives_list.append({
            'id': item[0],
            'name': item[1],
            'img': base64.b64encode(item[2]).decode('utf-8'),
            'relation': item[3],
            'description': item[4],
            'audio': base64.b64encode(item[5]).decode('utf-8'),
            'patient_id': item[6],
            'mobile': item[7]
        })
    return render_template('admin/manage_realatives.html', patient_id = patient_id, relatives_data = relatives_list, message = message)



@app.route('/delete_relative/<int:relative_id>/<int:patient_id>')
def delete_relative(relative_id, patient_id):

    query = "DELETE FROM relatives WHERE id = %s"
    values = (relative_id,)
    executionquery(query, values)

    query = "SELECT * FROM relatives WHERE patient_id = %s"
    values = (patient_id, )
    relatives_data = retrivequery1(query, values)

    relatives_list = []
    for item in relatives_data:
        relatives_list.append({
            'id': item[0],
            'name': item[1],
            'img': base64.b64encode(item[2]).decode('utf-8'),
            'relation': item[3],
            'description': item[4],
            'audio': base64.b64encode(item[5]).decode('utf-8'),
            'patient_id': item[6],
            'mobile': item[7]
        })
    return render_template('admin/manage_realatives.html', patient_id = patient_id, relatives_data = relatives_list, message = "Deleted successfully!")




@app.route('/prediction/<patient_id>', methods = ["GET", "POST"])
def prediction(patient_id):
    if request.method == "POST":
        myfile = request.files['img']
        fn = myfile.filename
        mypath = os.path.join(r'static\images\Saved_images', fn)
        myfile.save(mypath)
        

        # Define class labels based on your training
        class_labels = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

        # Load the saved model
        model = tf.keras.models.load_model(r"Models\mobilenet_model.h5")

        # Function to preprocess image and make prediction
        def predict_class(image_path):
            # Load the image, resize it to 224x224 (the target size used in training)
            img = image.load_img(image_path, target_size=(224, 224), color_mode='grayscale')
            
            # Convert the image to a numpy array and expand dimensions (for batch size of 1)
            img_array = image.img_to_array(img)  # Convert image to numpy array
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            
            # Normalize the image
            img_array /= 255.0  # Same normalization as during training
            
            # Predict using the model
            predictions = model.predict(img_array)
            
            # Get the class index with the highest probability
            predicted_class_idx = np.argmax(predictions, axis=1)[0]
            
            # Get the class label and confidence
            predicted_class_label = class_labels[predicted_class_idx]
            confidence = predictions[0][predicted_class_idx]
            
            return predicted_class_label, confidence

        # Example usage
        image_path = mypath
        predicted_class_label, confidence = predict_class(image_path)

        return render_template('admin/prediction.html', patient_id = patient_id, prediction = predicted_class_label, path = mypath)
    return render_template('admin/prediction.html', patient_id = patient_id)



@app.route('/description', methods = ["POST"])
def description():
    img_path = request.form['img']  
    condition = request.form['condition']
    description = request.form['description']
    patient_id = request.form['patient_id']

    if os.path.exists(img_path):
        with open(img_path, 'rb') as img_file:
            binary_data = img_file.read()

        query = "INSERT INTO patient_condition (img, `condition`, description, patient_id) VALUES (%s, %s, %s, %s)"
        values = (binary_data, condition, description, patient_id)
        executionquery(query, values)

        os.remove(img_path)

        return render_template('admin/prediction.html', patient_id = patient_id, message = "description sended successfully!")

    else:
        return render_template('admin/prediction.html', patient_id = patient_id, message = "Image not found!")



if __name__ == '__main__':
    app.run(debug = True)
