from flask_mysqldb import MySQL
from functools32 import wraps
from passlib.handlers.sha2_crypt import sha256_crypt

from DBconnect import connection
from IPython.core.display import display
from flask import Flask, render_template, request, jsonify, json, session, escape, redirect, url_for, flash
from hashlib import md5
import gc
import requests
import dt
from flask_wtf import Form
from wtforms import BooleanField, TextField, PasswordField, DateField, validators
import sys
import datetime

import pandas as pd
import math
import numpy
from scipy.stats import skew, kurtosis
from statsmodels.tsa import stattools

import os
import csv
from cStringIO import StringIO
import glob

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
from pyAudioAnalysis import audioSegmentation as aS
from pydub import AudioSegment

from ftplib import FTP

from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.cross_validation import train_test_split

app = Flask(__name__)
app.debug = True
app.secret_key = 'quiet!'


myPath = ""


@app.route("/")
def main():
    if 'logged_in' in session:
        return render_template('index.html', session_user_name=session['username'])
    return redirect(url_for('showSignIn'))
    #return render_template('index.html')


@app.route('/showSignUp', methods=["GET", "POST"])
def showSignUp():
    try:
        if request.method == "POST":
            name = request.form['inputName']
            email = request.form['inputEmail']
            password = sha256_crypt.encrypt(str(request.form['inputPassword']))
            c, conn = connection()

            x = c.execute("SELECT * FROM doctor WHERE docname = (%s)",
                          name)

            today = datetime.datetime.today().strftime("%m/%d/%Y")
            if int(x) > 0:
                flash("That username is already taken, please choose another")
                return render_template('signup.html')

            else:
                c.execute("INSERT INTO doctor (doc_id, docname, email, password, created_at) VALUES (%s, %s, %s, %s, %s)",
                          (name, name, email, password, today))

                conn.commit()
                flash("Thanks for registering!")
                c.close()
                conn.close()
                gc.collect()

                session['logged_in'] = True
                session['username'] = name

                return redirect(url_for('main'))

        return render_template("signup.html")

    except Exception as e:
        return (str(e))
    return render_template('signup.html')


def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash("You need to login first")
            return redirect(url_for('showSignIn'))
    return wrap


@app.route('/showSignIn', methods=["GET", "POST"])
def showSignIn():
    error = ''
    try:
        if request.method == "POST":
            name = request.form['inputName']
            password = sha256_crypt.encrypt(str(request.form['inputPassword']))
            c, conn = connection()

            data = c.execute("SELECT * FROM doctor WHERE docname = (%s)",
                             name)

            data = c.fetchone()[4]

            if sha256_crypt.verify(password, data):
                session['logged_in'] = True
                session['username'] = name

                flash("You are now logged in")
                return redirect(url_for("main"))

            else:
                print data
                error = "Invalid credentials, try again."

        gc.collect()

        return render_template("signin.html", error=error)

    except Exception as e:
        # flash(e)
        error = "Invalid credentials, try again."
        return render_template("signin.html", error=error)


@app.route('/logout')
@login_required
def logout():
    session.clear()
    flash("You have been logged out!")
    gc.collect()
    return redirect(url_for('main'))

class DateForm(Form):
    dt = DateField('Pick a Date', format="%m/%d/%Y")

class RegistrationForm(Form):
    username = TextField('Username', [validators.Length(min=4, max=20)])
    email = TextField('Email Address', [validators.Length(min=6, max=50)])
    password = PasswordField('New Password', [
        validators.DataRequired(),
        validators.EqualTo('confirm', message='Passwords must match')
    ])
    confirm = PasswordField('Repeat Password')
    accept_tos = BooleanField('I accept the Terms of Service and Privacy Notice (updated Jan 22, 2015)', [validators.DataRequired()])

@app.route('/get_users')
def get_users():
    # Query DB:
    cur, conn = connection()
    q_list_one = "SELECT * FROM users"
    cur.execute(q_list_one)
    users = cur.fetchall()

    users_dict = []
    for user in users:
        user_dict = {
            'Id': user[0],
            'IdUniq': user[1],
            'Name': user[2],
            'Age': user[3],
            'Date': user[4],
            'Imei': user[5],}
        users_dict.append(user_dict)

    return json.dumps(users_dict)


@app.route('/getUID', methods=["GET", "POST"])
def getUID():
    userid = request.form.get('userid')
    return userid
    # return render_template('getUID.html', userid=userid)

# @app.route('/getStartDate', methods=["GET", 'POST'])
# def getStartDate():
#     startdate = request.form.get('startdate')
#     return startdate
#
# @app.route('/getEndDate', methods=["GET", 'POST'])
# def getEndDate():
#     enddate = request.form.get('enddate')
#     return enddate

@app.route("/extractFeatures", methods=["GET", "POST"])
def extract():
    global userid
    userid = ""
    if request.method == 'POST':
        #userid = request.form['userid']
        userid = request.args.get('userid')
    #print userid
    return userid
    #return redirect(url_for('dashboard', userid=userid))



@app.route("/extractCSVFeatures", methods=["GET", "POST"])
def extractCSV():
    global myPath, userid
    # Declarations
    userid = ""
    myPath = ""
    jsonFeatures = {}

    if request.method == "POST":
        #userid = request.form['userid']
        userid = request.args.get('userid')

        print " Android must Have invoked this CSV shit"

        # Ftp connection
        FtpHostName = "anis.tunisia-webhosting.com"
        FtpUser = "ahmed@anis.tunisia-webhosting.com"
        FtpPassword = "ahmedahmed"
        ftp = FTP(FtpHostName)
        ftp.login(FtpUser, FtpPassword)
        path = "http://www.anis.tunisia-webhosting.com/anis.tunisia-webhosting.com/ahmed"

    ########################################
    ########################################

    # Accelerometer CSV data manipulation and features extraction

    #########################################
    #########################################
    #########################################


    # SQL queries
    #con = mdb.connect(hostname, username, password, database)

        curCSV, conn = connection()
        reqCSV = "SELECT * FROM files AS f WHERE f.uid = %s and f.type = 'CSV' "
        curCSV.execute(reqCSV, userid)
        CSVfiles = curCSV.fetchall()

        def magnitude(activity):
            x2 = activity['xAxis'] * activity['xAxis']
            y2 = activity['yAxis'] * activity['yAxis']
            z2 = activity['zAxis'] * activity['zAxis']
            m2 = x2 + y2 + z2
            m = m2.apply(lambda x: math.sqrt(x))
            return m

        def windows(df, size=100):
            start = 0
            while start < df.count():
                yield start, start + size
                start += (size / 2)

        def jitter(axis, start, end):
            j = float(0)
            for i in xrange(start, min(end, axis.count())):
                if start != 0:
                    j += abs(axis[i] - axis[i - 1])
            return j / (end - start)

        def mean_crossing_rate(axis, start, end):
            cr = 0
            m = axis.mean()
            for i in xrange(start, min(end, axis.count())):
                if start != 0:
                    p = axis[i - 1] > m
                    c = axis[i] > m
                    if p != c:
                        cr += 1
            return float(cr) / (end - start - 1)

        def window_summary(axis, start, end):
            acf = stattools.acf(axis[start:end])
            acv = stattools.acovf(axis[start:end])
            sqd_error = (axis[start:end] - axis[start:end].mean()) ** 2
            return [
                jitter(axis, start, end),
                mean_crossing_rate(axis, start, end),
                axis[start:end].mean(),
                axis[start:end].std(),
                axis[start:end].var(),
                axis[start:end].min(),
                axis[start:end].max(),
                acf.mean(),  # mean auto correlation
                acf.std(),  # standard deviation auto correlation
                acv.mean(),  # mean auto covariance
                acv.std(),  # standard deviation auto covariance
                skew(axis[start:end]),
                kurtosis(axis[start:end]),
                math.sqrt(sqd_error.mean())
            ]

        def features(activity):
            for (start, end) in windows(activity['timestamp']):
                features = []
                for axis in ['xAxis', 'yAxis', 'zAxis', 'magnitude']:
                    features += window_summary(activity[axis], start, end)
                yield features

        COLUMNS = ['timestamp', 'xAxis', 'yAxis', 'zAxis']
        features_dict = []
        for csvfile in CSVfiles:
            Activity = pd.read_csv(path + csvfile[1], header=None, names=COLUMNS)[:3000]
            Activity['magnitude'] = magnitude(Activity)

            with open('/home/ahmed/Desktop/flaskTherapio/AccFtr/' + csvfile[1][17:] + '_Features.csv', 'w') as out:
                rows = csv.writer(out)
                for f in features(Activity):
                    rows.writerow(f)

            ActivityDataFeature = pd.read_csv('/home/ahmed/Desktop/flaskTherapio/AccFtr/' + csvfile[1][17:] +
                                          '_Features.csv', header=None)

            # return ActivityDataFeature.reset_index().to_json(orient='index')
            # CSVDATA = json.dumps(csv_files_dict)

        return "Shit must be invoked by android and the user id is   " + str(userid)



@app.route("/extractAudioFeatures", methods=["GET", "POST"])
def extractAudio():

    # Declarations
    global userid, starttime, endtime, myPath, numbPauses,loudness,dur,madmax,maxDBFS,MPA,numOfFtrs,phoneCallFtrFinal,wavo,jsonFeatures,nf
    userid = ""
    starttime =""
    endtime = ""
    myPath = ""
    numbPauses = 0
    loudness = 0
    nf = 0
    dur = 0
    madmax = 0
    maxDBFS = 0
    MPA = 0
    numOfFtrs = 0
    wavo = ""
    jsonFeatures = {}
    phoneCallFtrFinal = pd.DataFrame()


    # getting the parameters
    if request.method == 'POST':
        userid = request.args.get('userid')
        #userid = request.form['userid']
        print " Android must Have invoked this"

        # Ftp connection
        FtpHostName = "anis.tunisia-webhosting.com"
        FtpUser = "ahmed@anis.tunisia-webhosting.com"
        FtpPassword = "ahmedahmed"
        ftp = FTP(FtpHostName)
        ftp.login(FtpUser, FtpPassword)
        path = "http://www.anis.tunisia-webhosting.com/anis.tunisia-webhosting.com/ahmed"

    ########################################
    ########################################

    # Phone calls processing and features extraction

    #########################################
    #########################################
    #########################################

    # SQL queries
        curWAV, conn = connection()
        reqWAV = "SELECT * FROM files AS f WHERE f.uid = %s and f.type = 'WAV' "
        curWAV.execute(reqWAV, userid)
        WAVfiles = curWAV.fetchall()

        wavs_dict = []
        for wavf in WAVfiles:
            wavf_dict = {
                'Id': wavf[0],
                'filepath': wavf[1],
                'type': wavf[2],
                'createdat': wavf[3],
                'uid': wavf[4]}
            wavs_dict.append(wavf_dict)

        #print wavs_dict.__len__()
        print len(WAVfiles)
        ind = 0
        patho = "/home/ahmed/Desktop/flaskTherapio/"
        for wavfile in WAVfiles:
            ind += 1
            print "checking database for the "+ str(ind)+ " time"
            dirPath = wavfile[1][1:17]
            wavName = wavfile[1][17:]
            wavdate = wavfile[3][1:10]
            #print dirPath
            print wavName
            print wavdate
            ftp.cwd("/"+dirPath)
            print "done changing directory"
            audiofilematch = '*.wav'
            cc = 0
            for filename in ftp.nlst(audiofilematch):  # Loop - looking for WAV files
                cc +=1
                print "checking FTP for the " + str(cc) + " time"
                print "comparing " + wavName + " and     "+ filename
                print wavName == filename
                fhandle = open(filename, 'wb')
                print 'Getting ' + filename
                os.chdir("/home/ahmed/Desktop/flaskTherapio/")
                ftp.retrbinary('RETR ' + filename, fhandle.write)
                print "stored"
                fhandle.close()
                if filename == wavName:
                    print " just do this shit here"
                    sound = AudioSegment.from_file(patho+filename)
                    sound.export(patho+"PhoneCallsFtr/"+filename+".wav",format="wav")
                    loudness = sound.rms
                    #number of Frames
                    nf = sound.frame_count()
                    # Value of loudness
                    loudness = sound.rms
                    #duration
                    dur = sound.duration_seconds
                     #max
                    madmax = sound.max
                    #max possible amplitude
                    MPA = sound.max_possible_amplitude
                    #max dbfs
                    maxDBFS = sound.max_dBFS
                    samplewidth = sound.sample_width

                    today = datetime.datetime.today().strftime("%m/%d/%Y")
                    blahblah = patho+"PhoneCallsFtr/"+filename+".wav"
                    print "Blah blah path :   ======>  " + blahblah


                    [Fs, x] = audioBasicIO.readAudioFile(blahblah)
                    #silence removal ---- into segments

                    segments = aS.silenceRemoval(x, Fs, 0.030, 0.030, smoothWindow = 0.3, Weight = 0.6, plot=True)
                    numbPauses = len(segments) - 1
                    if numbPauses:
                        error = False
                    else:
                        error = True
                    print "num pauses : " + str(numbPauses)
                    # SELECT * FROM `phoneCallFeatures` WHERE (`created_at` BETWEEN 60 AND 1500) AND (uid = '5795028d168257.04609170')
                    # SQL query to insert phone call features to DB
                    insertFtrReq = """INSERT INTO phoneCallFeatures
                                              (uid,
                                              npause,
                                              loudness,
                                              maxA,
                                              created_at)
                                              VALUES (%s,
                                                      %s,
                                                      %s,
                                                      %s,
                                                      %s)""", (userid,
                                                               numbPauses,
                                                               loudness,
                                                               madmax,
                                                               wavdate)
                    curInsertFtr, conn = connection()
                    curInsertFtr.execute(*insertFtrReq)
                    conn.commit()


                    # short term features extraction
                    fw = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.1*Fs, 0.1*Fs)
                    print fw
                    # Mid term feature extraction
                    #F1 = audioFeatureExtraction.mtFeatureExtraction(x, Fs, 0.050*Fs, 0.050*Fs,0.050*Fs, 0.050*Fs)
                    mypath = "/home/ahmed/Desktop/flaskTherapio/PhoneCallsFtr/"
                    if not os.path.isdir(mypath):
                        os.makedirs(mypath)
                    numpy.savetxt(mypath+wavName+"_AllStFtrs.csv", fw, delimiter=",")
                    #numpy.savetxt("00_AllMidFtrs.csv", F1, delimiter=",")
                    # read features in a data frame
                    phoneCallFtrPrime = pd.read_csv(mypath +wavName+"_AllStFtrs.csv")
                    COLUMNS = []
                    for i in range(0, len(phoneCallFtrPrime.columns)):
                        COLUMNS.append('Frame' + str(i + 1))
                        COLUMNS.append('label')
                    phoneCallFtrFinal = pd.read_csv(mypath +wavName+"_AllStFtrs.csv", header=None, names=COLUMNS)

                    numOfFtrs = len(phoneCallFtrFinal["label"])
                    phoneCallFtrFinal.label = phoneCallFtrFinal.label.astype(float).fillna(0)
                    # phoneCallFtrFinal.label = phoneCallFtrFinal.label.astype(int).fillna(0)
                    for j in range(0, numOfFtrs):
                        phoneCallFtrFinal.iloc[j]['label'] = 0

                    #jsonFeatures = phoneCallFtrFinal.reset_index().to_json(orient='index')
                    #jsonFeatures = phoneCallFtrFinal.to_json(orient=None)
                    display(phoneCallFtrFinal.head())
                    #print "JSON Features of every audio ----------------------------------------------"
                    #print jsonFeatures

    print "this script has been called and data has been handled"
    return jsonify({'USER ID': userid}), 201



@app.route("/dashboard/<userid>", methods=["GET", "POST"])
def dashboard(userid):

    # Declarations
    global starttime, endtime, myPath, numbPauses,loudness,dur,madmax,maxDBFS,MPA,numOfFtrs,phoneCallFtrFinal,wavo,jsonFeatures,nf
    starttime =""
    endtime = ""
    myPath = ""
    numbPauses = 0
    loudness = 0
    nf = 0
    dur = 0
    madmax = 0
    maxDBFS = 0
    MPA = 0
    numOfFtrs = 0
    wavo = ""
    jsonFeatures = {}
    phoneCallFtrFinal = pd.DataFrame()

    # Ftp connection
    FtpHostName = "anis.tunisia-webhosting.com"
    FtpUser = "ahmed@anis.tunisia-webhosting.com"
    FtpPassword = "ahmedahmed"
    ftp = FTP(FtpHostName)
    ftp.login(FtpUser, FtpPassword)
    path = "http://www.anis.tunisia-webhosting.com/anis.tunisia-webhosting.com/ahmed"

    # forms
    form = DateForm()
    if form.validate_on_submit():
        starttime = form.dt.data.strftime('%x')
        #endtime = form.dt.data.strftime('%x')


        ########################################
        ########################################

        # Accelerometer CSV data manipulation and features extraction

        #########################################
        #########################################
        #########################################


    # SQL queries
    curCSV, conn = connection()
    reqCSV = "SELECT * FROM files AS f WHERE f.uid = %s and f.type = 'CSV' "
    curCSV.execute(reqCSV, userid)
    CSVfiles = curCSV.fetchall()

    def magnitude(activity):
        x2 = activity['xAxis'] * activity['xAxis']
        y2 = activity['yAxis'] * activity['yAxis']
        z2 = activity['zAxis'] * activity['zAxis']
        m2 = x2 + y2 + z2
        m = m2.apply(lambda x: math.sqrt(x))
        return m

    def windows(df, size=100):
        start = 0
        while start < df.count():
            yield start, start + size
            start += (size / 2)

    def jitter(axis, start, end):
        j = float(0)
        for i in xrange(start, min(end, axis.count())):
            if start != 0:
                j += abs(axis[i] - axis[i - 1])
        return j / (end - start)

    def mean_crossing_rate(axis, start, end):
        cr = 0
        m = axis.mean()
        for i in xrange(start, min(end, axis.count())):
            if start != 0:
                p = axis[i - 1] > m
                c = axis[i] > m
                if p != c:
                    cr += 1
        return float(cr) / (end - start - 1)

    def window_summary(axis, start, end):
        acf = stattools.acf(axis[start:end])
        acv = stattools.acovf(axis[start:end])
        sqd_error = (axis[start:end] - axis[start:end].mean()) ** 2
        return [
            jitter(axis, start, end),
            mean_crossing_rate(axis, start, end),
            axis[start:end].mean(),
            axis[start:end].std(),
            axis[start:end].var(),
            axis[start:end].min(),
            axis[start:end].max(),
            acf.mean(),  # mean auto correlation
            acf.std(),  # standard deviation auto correlation
            acv.mean(),  # mean auto covariance
            acv.std(),  # standard deviation auto covariance
            skew(axis[start:end]),
            kurtosis(axis[start:end]),
            math.sqrt(sqd_error.mean())
        ]

    def features(activity):
        for (start, end) in windows(activity['timestamp']):
            features = []
            for axis in ['xAxis', 'yAxis', 'zAxis', 'magnitude']:
                features += window_summary(activity[axis], start, end)
            yield features

    COLUMNS = ['timestamp', 'xAxis', 'yAxis', 'zAxis']
    features_dict = []
    for csvfile in CSVfiles:
        Activity = pd.read_csv(path + csvfile[1], header=None, names=COLUMNS)[:3000]
        Activity['magnitude'] = magnitude(Activity)

        with open('/home/ahmed/Desktop/flaskTherapio/AccFtr/' + csvfile[1][17:] + '_Features.csv', 'w') as out:
            rows = csv.writer(out)
            for f in features(Activity):
                rows.writerow(f)

        ActivityDataFeature = pd.read_csv('/home/ahmed/Desktop/flaskTherapio/AccFtr/' + csvfile[1][17:] +
                                          '_Features.csv', header=None)

    # return ActivityDataFeature.reset_index().to_json(orient='index')
    # CSVDATA = json.dumps(csv_files_dict)


    ########################################
    ########################################

    # Phone calls processing and features extraction

    #########################################
    #########################################
    #########################################

    curWAV, conn = connection()
    reqWAV = "SELECT * FROM files AS f WHERE f.uid = %s and f.type = 'WAV' "
    curWAV.execute(reqWAV, userid)
    WAVfiles = curWAV.fetchall()

    wavs_dict = []
    for wavf in WAVfiles:
        wavf_dict = {
            'Id': wavf[0],
            'filepath': wavf[1],
            'type': wavf[2],
            'createdat': wavf[3],
            'uid': wavf[4]}
        wavs_dict.append(wavf_dict)

    #print wavs_dict.__len__()
    print len(WAVfiles)
    ind = 0
    patho = "/home/ahmed/Desktop/flaskTherapio/"
    for wavfile in WAVfiles:
        ind += 1
        print "checking database for the "+ str(ind)+ " time"
        dirPath = wavfile[1][1:17]
        wavName = wavfile[1][17:]
        wavdate = wavfile[3][1:10]
        #print dirPath
        print wavName
        print wavdate
        ftp.cwd("/"+dirPath)
        print "done changing directory"
        audiofilematch = '*.wav'
        cc = 0
        error = False
        for filename in ftp.nlst(audiofilematch):  # Loop - looking for WAV files
            if error == False:

                cc +=1
                print "checking FTP for the " + str(cc) + " time"
                print "comparing " + wavName + " and     "+ filename
                print wavName == filename
                fhandle = open(filename, 'wb')
                print 'Getting ' + filename
                os.chdir("/home/ahmed/Desktop/flaskTherapio/")
                ftp.retrbinary('RETR ' + filename, fhandle.write)
                print "stored"
                fhandle.close()
                if filename == wavName:
                    print " just do this shit here"
                    sound = AudioSegment.from_file(patho+filename)
                    sound.export(patho+"PhoneCallsFtr/"+filename+".wav",format="wav")
                    loudness = sound.rms
                    #number of Frames
                    nf = sound.frame_count()
                    # Value of loudness
                    loudness = sound.rms
                    #duration
                    dur = sound.duration_seconds
                    #max
                    madmax = sound.max
                    #max possible amplitude
                    MPA = sound.max_possible_amplitude
                    #max dbfs
                    maxDBFS = sound.max_dBFS
                    samplewidth = sound.sample_width

                    today = datetime.datetime.today().strftime("%m/%d/%Y")
                    blahblah = patho+"PhoneCallsFtr/"+filename+".wav"
                    print "Blah blah path :   ======>  " + blahblah


                    [Fs, x] = audioBasicIO.readAudioFile(blahblah)
                    #silence removal ---- into segments

                    segments = aS.silenceRemoval(x, Fs, 0.030, 0.030, smoothWindow = 0.3, Weight = 0.6, plot=True)
                    numbPauses = len(segments) - 1
                    if numbPauses:
                        error = False
                    else:
                        error = True
                    print "num pauses : " + str(numbPauses)
                    # SELECT * FROM `phoneCallFeatures` WHERE (`created_at` BETWEEN 60 AND 1500) AND (uid = '5795028d168257.04609170')
                    # SQL query to insert phone call features to DB
                    insertFtrReq = """INSERT INTO phoneCallFeatures
                                              (uid,
                                              npause,
                                              loudness,
                                              maxA,
                                              created_at)
                                              VALUES (%s,
                                                      %s,
                                                      %s,
                                                      %s,
                                                      %s)""", (userid,
                                                               numbPauses,
                                                               loudness,
                                                               madmax,
                                                               wavdate)
                    curInsertFtr, conn = connection()
                    curInsertFtr.execute(*insertFtrReq)
                    conn.commit()


                    # short term features extraction
                    fw = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.1*Fs, 0.1*Fs)
                    print fw
                    # Mid term feature extraction
                    #F1 = audioFeatureExtraction.mtFeatureExtraction(x, Fs, 0.050*Fs, 0.050*Fs,0.050*Fs, 0.050*Fs)
                    mypath = "/home/ahmed/Desktop/flaskTherapio/PhoneCallsFtr/"
                    if not os.path.isdir(mypath):
                        os.makedirs(mypath)
                    numpy.savetxt(mypath+wavName+"_AllStFtrs.csv", fw, delimiter=",")
                    #numpy.savetxt("00_AllMidFtrs.csv", F1, delimiter=",")
                    # read features in a data frame
                    phoneCallFtrPrime = pd.read_csv(mypath +wavName+"_AllStFtrs.csv")
                    COLUMNS = []
                    for i in range(0, len(phoneCallFtrPrime.columns)):
                        COLUMNS.append('Frame' + str(i + 1))
                        COLUMNS.append('label')
                    phoneCallFtrFinal = pd.read_csv(mypath +wavName+"_AllStFtrs.csv", header=None, names=COLUMNS)

                    numOfFtrs = len(phoneCallFtrFinal["label"])
                    phoneCallFtrFinal.label = phoneCallFtrFinal.label.astype(float).fillna(0)
                    # phoneCallFtrFinal.label = phoneCallFtrFinal.label.astype(int).fillna(0)
                    for j in range(0, numOfFtrs):
                        phoneCallFtrFinal.iloc[j]['label'] = 0

                #jsonFeatures = phoneCallFtrFinal.reset_index().to_json(orient='index')
                #jsonFeatures = phoneCallFtrFinal.to_json(orient=None)
                    display(phoneCallFtrFinal.head())
                #print "JSON Features of every audio ----------------------------------------------"
                #print jsonFeatures
            else:
                print " Moving to the next file"

    curPhone, conn = connection()
    reqPhoneCallFtr = """SELECT * FROM phoneCallFeatures WHERE
                          created_at BETWEEN  %s AND %s
                          AND uid = %s""",(starttime, today, userid)

    curPhone.execute(*reqPhoneCallFtr)
    calls = curPhone.fetchall()

    calls_dict = []
    for call in calls:
        call_dict = {
            'Id': call[0],
            'IdUniq': call[1],
            'npause': call[2],
            'loudness': call[3],
            'maxA': call[4],
            'createdat': call[5],}
        calls_dict.append(call_dict)
    #callJsonData = json.dumps(calls_dict)
    #LoadJsonData = json.loads(callJsonData)
    #print "JSON calls data ----------------------------------------------"
    #print LoadJsonData
    #print LoadJsonData['loudness']

    return render_template('dashboard.html', userid=userid, loudness=loudness, nf=nf, dur=dur, maxDBFS=maxDBFS,
                           madmax=madmax, MPA=MPA, jsonFeatures=jsonFeatures,starttime=starttime, form=form)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9000)
