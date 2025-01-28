import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import \
    tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, render_template, url_for, request
import sqlite3
import cv2
import shutil
import csv



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return render_template('patientdetails.html')

    return render_template('userlog.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')



@app.route('/adminlog', methods=['GET', 'POST'])
def adminlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if result:
            f = open('Patientdetails.csv', 'r')
            reader = csv.reader(f)
            result1 = []
            for row in reader:
                result1.append(row)
            f.close() 

            if result1:
                return render_template('adminlog.html', result1=result1)
            else:
                return render_template('adminlog.html', msg='data not found')
        else:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
    return render_template('index.html')


@app.route('/adminreg', methods=['GET', 'POST'])
def adminreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()
        
        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')




@app.route('/create_datsets',  methods=['POST','GET'])
def create_datsets():
    if request.method == 'POST':
        Id = request.form['Id']
        Name = request.form['Name']
        Phone = request.form['Phone']
        Email = request.form['Email']
        Gender = request.form['Gender']
        Age = request.form['Age']
        Address = request.form['Address']
        #cardnum = request.form['cardnum']

        print(Id+' '+Name+' '+Phone+' '+Email+' '+Gender+' '+Age+' '+Address)

        row = [Id, Name, Phone, Email, Gender, Age, Address,]




        if not os.path.exists('Patientdetails.csv'):
            row1 = ['Id', 'Name', 'Phone', 'Email', 'Gender', 'Age', 'Adress']
            with open('Patientdetails.csv','w',newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row1)
            csvFile.close()

        with open('Patientdetails.csv','a', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
           
        csvFile.close()


        return render_template("userlog.html")



@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
 
        dirPath = "static/images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        fileName=request.form['filename']
        dst = "static/images"
        

        shutil.copy("test/"+fileName, dst)
        image = cv2.imread("test/"+fileName)
        
        #color conversion
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('static/gray.jpg', gray_image)
        #apply the Canny edge detection
        edges = cv2.Canny(image, 250, 254)
        cv2.imwrite('static/edges.jpg', edges)
        #apply thresholding to segment the image
        retval2,threshold2 = cv2.threshold(gray_image,128,255,cv2.THRESH_BINARY)
        cv2.imwrite('static/threshold.jpg', threshold2)
        # create the sharpening kernel
        kernel_sharpening = np.array([[-1,-1,-1],
                                    [-1, 9,-1],
                                    [-1,-1,-1]])

        # apply the sharpening kernel to the image
        sharpened = cv2.filter2D(image, -1, kernel_sharpening)

        # save the sharpened image
        cv2.imwrite('static/sharpened.jpg', sharpened)
        
        verify_dir = 'static/images'
        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'cervicalcancer-{}-{}.model'.format(LR, '2conv-basic')
    ##    MODEL_NAME='keras_model.h5'
        def process_verify_data():
            verifying_data = []
            for img in os.listdir(verify_dir):
                path = os.path.join(verify_dir, img)
                img_num = img.split('.')[0]
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                verifying_data.append([np.array(img), img_num])
                np.save('verify_data.npy', verifying_data)
            return verifying_data

        verify_data = process_verify_data()
        #verify_data = np.load('verify_data.npy')

        
        tf.compat.v1.reset_default_graph()
        #tf.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 11, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('model loaded!')


        fig = plt.figure()
        
        str_label=" "
        accuracy=""
        for num, data in enumerate(verify_data):

            img_num = data[1]
            img_data = data[0]

            y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            # model_out = model.predict([data])[0]
            model_out = model.predict([data])[0]
            print(model_out)
            print('model {}'.format(np.argmax(model_out)))

            

            if np.argmax(model_out) == 0:
                str_label = 'stage1(1A)'
                print("The predicted image of the stage1(1A) is with a accuracy of {} %".format(model_out[0]*100))
                accuracy = "The predicted image of the stage1(1A) is with a accuracy of {} %".format(model_out[0]*100)
                pre="The Precaution for stage1 cancer"
                pre1=["Regular Pap smears: Routine Pap smears can detect abnormal cervical cells early, allowing for timely intervention and treatment",
                "HPV vaccination: Vaccination against human papillomavirus (HPV) can prevent infection with high-risk strains linked to cervical cancer.",
                "Avoidance of tobacco: Smoking increases the risk of cervical cancer, so quitting smoking or avoiding exposure to secondhand smoke is crucial"]
                
            elif np.argmax(model_out) == 1:
                str_label = 'stage1(11A)'
                print("The predicted image of the stage1(11A) is with a accuracy of {} %".format(model_out[1]*100))
                accuracy = "The predicted image of the stage1(11A) is with a accuracy of {} %".format(model_out[1]*100)
                pre="The precaution for stage1 cancer"
                pre1=["Regular Pap smears: Routine Pap smears can detect abnormal cervical cells early, allowing for timely intervention and treatment",
                "HPV vaccination: Vaccination against human papillomavirus (HPV) can prevent infection with high-risk strains linked to cervical cancer.",
                "Avoidance of tobacco: Smoking increases the risk of cervical cancer, so quitting smoking or avoiding exposure to secondhand smoke is crucial"]
                
            elif np.argmax(model_out) == 2:
                str_label = 'stage2(1B)'
                print("The predicted image of the stage2(1B) is with a accuracy of {} %".format(model_out[2]*100))
                accuracy = "The predicted image of the stage2(1B) is with a accuracy of {} %".format(model_out[2]*100)
                pre="The precaution for stage2 cancer"
                pre1=["Regular Pap tests: Schedule routine Pap tests to detect any abnormal changes in the cervix early on.",
                "HPV vaccination: Get vaccinated against Human Papillomavirus (HPV), as it's a leading cause of cervical cancer.",
                "Follow-up appointments: Attend scheduled follow-up appointments with your healthcare provider for monitoring and further evaluation.",
                "Healthy lifestyle: Maintain a healthy lifestyle by avoiding smoking, practicing safe sex, and eating a balanced diet rich in fruits and vegetables."]
                
            elif np.argmax(model_out) == 3:
                str_label = 'stage2(11B)'
                print("The predicted image of the stage2(11B) is with a accuracy of {} %".format(model_out[3]*100))
                accuracy = "The predicted image of the stage2(11B) is with a accuracy of {} %".format(model_out[3]*100)
                pre="The precaution for stage2 cancer"
                pre1=["Regular Pap tests: Schedule routine Pap tests to detect any abnormal changes in the cervix early on.",
                "HPV vaccination: Get vaccinated against Human Papillomavirus (HPV), as it's a leading cause of cervical cancer.",
                "Follow-up appointments: Attend scheduled follow-up appointments with your healthcare provider for monitoring and further evaluation.",
                "Healthy lifestyle: Maintain a healthy lifestyle by avoiding smoking, practicing safe sex, and eating a balanced diet rich in fruits and vegetables."]
                
            elif np.argmax(model_out) == 4:
                str_label = 'stage2(111B)'
                print("The predicted image of the stage2(111B) is with a accuracy of {} %".format(model_out[4]*100))
                accuracy = "The predicted image of the stage2(111B) is with a accuracy of {} %".format(model_out[4]*100)
                pre="The precaution for stage2 cancer"
                pre1=["Regular Pap tests: Schedule routine Pap tests to detect any abnormal changes in the cervix early on.",
                "HPV vaccination: Get vaccinated against Human Papillomavirus (HPV), as it's a leading cause of cervical cancer.",
                "Follow-up appointments: Attend scheduled follow-up appointments with your healthcare provider for monitoring and further evaluation.",
                "Healthy lifestyle: Maintain a healthy lifestyle by avoiding smoking, practicing safe sex, and eating a balanced diet rich in fruits and vegetables."]
                
            elif np.argmax(model_out) == 5:
                str_label = 'stage3(1C)'
                print("The predicted image of the stage3(1C) is with a accuracy of {} %".format(model_out[5]*100))
                accuracy = "The predicted image of the stage3(1C) is with a accuracy of {} %".format(model_out[5]*100)
                pre="The precaution for stage3 cancer"
                pre1=["Regular Pap smears or HPV tests for early detection.",
                "Consultation with a gynecologic oncologist for treatment planning.",
                "Adherence to prescribed treatment regimen, including surgery, chemotherapy, and/or radiation therapy.",
                "Lifestyle modifications such as quitting smoking and maintaining a healthy weight to reduce risk factors."]
                
            elif np.argmax(model_out) == 6:
                str_label = 'stage3(11C)'
                print("The predicted image of the stage3(11C) is with a accuracy of {} %".format(model_out[6]*100))
                accuracy = "The predicted image of the stage3(11C) is with a accuracy of {} %".format(model_out[6]*100)
                pre="The precaution for stage3 cancer"
                pre1=["Regular Pap smears or HPV tests for early detection.",
                "Consultation with a gynecologic oncologist for treatment planning.",
                "Adherence to prescribed treatment regimen, including surgery, chemotherapy, and/or radiation therapy.",
                "Lifestyle modifications such as quitting smoking and maintaining a healthy weight to reduce risk factors."]
                
            elif np.argmax(model_out) == 7:
                str_label = 'stage3(111C)'
                print("The predicted image of the stage3(111C) is with a accuracy of {} %".format(model_out[7]*100))
                accuracy = "The predicted image of the stage3(111C) is with a accuracy of {} %".format(model_out[7]*100)
                pre="The precaution for stage3 cancer"
                pre1=["Regular Pap smears or HPV tests for early detection.",
                "Consultation with a gynecologic oncologist for treatment planning.",
                "Adherence to prescribed treatment regimen, including surgery, chemotherapy, and/or radiation therapy.",
                "Lifestyle modifications such as quitting smoking and maintaining a healthy weight to reduce risk factors."]
                
            elif np.argmax(model_out) == 8:
                str_label = 'stage4(1D)'
                print("The predicted image of the stage4(1D) is with a accuracy of {} %".format(model_out[8]*100))
                accuracy = "The predicted image of the stage4(1D) is with a accuracy of {} %".format(model_out[8]*100)
                pre="The precaution for stage4 cancer"
                pre1=["Regular Pap tests: Schedule routine Pap tests to detect any abnormal changes in the cervix early on.",
                "HPV vaccination: Get vaccinated against Human Papillomavirus (HPV), as it's a leading cause of cervical cancer.",
                "Follow-up appointments: Attend scheduled follow-up appointments with your healthcare provider for monitoring and further evaluation.",
                "Healthy lifestyle: Maintain a healthy lifestyle by avoiding smoking, practicing safe sex, and eating a balanced diet rich in fruits and vegetables."]
                
            elif np.argmax(model_out) == 9:
                str_label = 'stage4(11D)'
                print("The predicted image of the stage4(11D) is with a accuracy of {} %".format(model_out[9]*100))
                accuracy = "The predicted image of the stage4(11D) is with a accuracy of {} %".format(model_out[9]*100)
                pre="The precaution for stage4 cancer"
                pre1=["Regular Pap tests: Schedule routine Pap tests to detect any abnormal changes in the cervix early on.",
                "HPV vaccination: Get vaccinated against Human Papillomavirus (HPV), as it's a leading cause of cervical cancer.",
                "Follow-up appointments: Attend scheduled follow-up appointments with your healthcare provider for monitoring and further evaluation.",
                "Healthy lifestyle: Maintain a healthy lifestyle by avoiding smoking, practicing safe sex, and eating a balanced diet rich in fruits and vegetables."]
                
            elif np.argmax(model_out) == 10:
                str_label = 'Normal'
                print("The predicted image of the Normal is with a accuracy of {} %".format(model_out[10]*100))
                accuracy = "The predicted image of the Normal is with a accuracy of {} %".format(model_out[10]*100)
               
            A=float(model_out[0])
            B=float(model_out[1])
            C=float(model_out[2])
            D=float(model_out[3])
            E=float(model_out[4])
            F=float(model_out[5])
            G=float(model_out[6])
            H=float(model_out[7])
            I=float(model_out[8])
            J=float(model_out[9])
            K=float(model_out[10])

            dic={'1A':A,'11A':B,'1B':C,'11B':D,'111B':E,'1C':F,'11C':G,'111C':H,'1D':I,'11D':J,'Normal':K}
            algm = list(dic.keys()) 
            accu = list(dic.values()) 
            fig = plt.figure(figsize = (5, 5))  
            plt.bar(algm, accu, color ='maroon', width = 0.3)  
            plt.xlabel("Comparision") 
            plt.ylabel("Accuracy Level") 
            plt.title("Accuracy Comparision between cervical cancer detection....")
            plt.savefig('static/matrix.png')
            
                            

        return render_template('results.html', status=str_label,accuracy=accuracy,precaution=pre,precaution1=pre1,ImageDisplay="http://127.0.0.1:5000/static/images/"+fileName,ImageDisplay1="http://127.0.0.1:5000/static/gray.jpg",ImageDisplay2="http://127.0.0.1:5000/static/edges.jpg",ImageDisplay3="http://127.0.0.1:5000/static/threshold.jpg",ImageDisplay4="http://127.0.0.1:5000/static/sharpened.jpg",ImageDisplay5="http://127.0.0.1:5000/static/matrix.png")
        
    return render_template('index.html')

@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
