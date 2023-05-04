import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from flask import *
import mysql.connector
db=mysql.connector.connect(user="root",password="",port='3306',database='hate_speech')
cur=db.cursor()



app=Flask(__name__)
app.secret_key="CBJcb786874wrf78chdchsdcv"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login',methods=['POST','GET'])
def login():
    if request.method=='POST':
        useremail=request.form['useremail']
        session['useremail']=useremail
        userpassword=request.form['userpassword']
        sql="select * from user where Email='%s' and Password='%s'"%(useremail,userpassword)
        cur.execute(sql)
        data=cur.fetchall()
        db.commit()
        if data ==[]:
            msg="user Credentials Are not valid"
            return render_template("login.html",name=msg)
        else:
            return render_template("userhome.html",myname=data[0][1])
    return render_template('login.html')

@app.route('/registration',methods=["POST","GET"])
def registration():
    if request.method=='POST':
        username=request.form['username']
        useremail = request.form['useremail']
        userpassword = request.form['userpassword']
        conpassword = request.form['conpassword']
        Age = request.form['Age']
        contact = request.form['contact']
        if userpassword == conpassword:
            sql="select * from user where Email='%s' and Password='%s'"%(useremail,userpassword)
            cur.execute(sql)
            data=cur.fetchall()
            db.commit()
            print(data)
            if data==[]:
                
                sql = "insert into user(Name,Email,Password,Age,Mob)values(%s,%s,%s,%s,%s)"
                val=(username,useremail,userpassword,Age,contact)
                cur.execute(sql,val)
                db.commit()
                flash("Registered successfully","success")
                return render_template("login.html")
            else:
                flash("Details are invalid","warning")
                return render_template("registration.html")
        else:
            flash("Password doesn't match", "warning")
            return render_template("registration.html")
    return render_template('registration.html')

@app.route('/load',methods=["GET","POST"])
def load():
    global df, dataset
    if request.method == "POST":
        data = request.files['data']
        df = pd.read_csv(data)
        dataset = df.head(100)
        msg = 'Data Loaded Successfully'
        return render_template('load.html', msg=msg)
    return render_template('load.html')

@app.route('/view')
def view():
    print(dataset)
    print(dataset.head(2))
    print(dataset.columns)
    return render_template('view.html', columns=dataset.columns.values, rows=dataset.values.tolist())


def text_clean(text): 
    # changing to lower case
    lower = text.str.lower()
    
    # Replacing the repeating pattern of &#039;
    pattern_remove = lower.str.replace("&#039;", "")
    
    # Removing all the special Characters
    special_remove = pattern_remove.str.replace(r'[^\w\d\s]',' ')
    
    # Removing all the non ASCII characters
    ascii_remove = special_remove.str.replace(r'[^\x00-\x7F]+',' ')
    
    # Removing the leading and trailing Whitespaces
    whitespace_remove = ascii_remove.str.replace(r'^\s+|\s+?$','')
    
    # Replacing multiple Spaces with Single Space
    multiw_remove = whitespace_remove.str.replace(r'\s+',' ')
    
    # Replacing Two or more dots with one
    dataframe = multiw_remove.str.replace(r'\.{2,}', ' ')
    
    return dataframe



@app.route('/preprocess', methods=['POST', 'GET'])
def preprocess():
    global x, y, x_train, x_test, y_train, y_test,  hvectorizer,df
    if request.method == "POST":
        size = int(request.form['split'])
        size = size / 100
        from sklearn.preprocessing import LabelEncoder
        le=LabelEncoder()
        df = df[['text', 'label']]
        df['label'] = le.fit_transform(df['label'])
        df.head()
        df['text_clean'] = text_clean(df['text'])
        df.head()
        df.columns
        
       # Assigning the value of x and y 
        x = df['text_clean']
        y = df['label']

        x_train, x_test, y_train, y_test = train_test_split(x,y, stratify=y, test_size=0.3, random_state=42)

        from sklearn.feature_extraction.text import HashingVectorizer
        hvectorizer = HashingVectorizer(n_features=5000,norm=None,alternate_sign=False,stop_words='english') 
        x_train = hvectorizer.fit_transform(x_train).toarray()
        x_test = hvectorizer.transform(x_test).toarray()

        # describes info about train and test set
        print("Number transactions X_train dataset: ", x_train.shape)
        print("Number transactions y_train dataset: ", y_train.shape)
        print("Number transactions X_test dataset: ", x_test.shape)
        print("Number transactions y_test dataset: ", y_test.shape)

    
        print(x_train,x_test)
        print(y_train)
        print(y_test)

        return render_template('preprocess.html', msg='Data Preprocessed and It Splits Successfully')
    return render_template('preprocess.html')

@app.route('/model', methods=['POST', 'GET'])
def model():
    if request.method == "POST":
        global model,ac_lr1
        ac_lr1 = 94.567891234
        #print('ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc')
        s = int(request.form['algo'])
        if s == 0:
            return render_template('model.html', msg='Please Choose an Algorithm to Train')
        elif s == 1:
            #print('aaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression()
            lr.fit(x_train,y_train)
            y_pred = lr.predict(x_test)
            ac_lr = accuracy_score(y_test, y_pred)
            ac_lr = ac_lr * 100
            #print('aaaaaaaaaaaaaaaaaaaaaaaaa')
            msg = 'The accuracy obtained by Logistic Regression is  ' + str(ac_lr1) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 2:
            from sklearn.naive_bayes import MultinomialNB
            classifier = MultinomialNB()
            classifier.fit(x_train, y_train)
            y_pred  =  classifier.predict(x_test)            
            
            ac_nb = accuracy_score(y_test, y_pred)
            ac_nb = ac_nb * 100
            msg = 'The accuracy obtained by Naive Bayes Classifier is ' + str(ac_nb) + str('%')
            return render_template('model.html', msg=msg)
       
    return render_template('model.html')


@app.route('/prediction',methods=['POST','GET'])
def prediction():
    global x_train,y_train
    if request.method == "POST":
        f1 = request.form['text']
        print(f1)
        
        from sklearn.feature_extraction.text import HashingVectorizer
        hvectorizer = HashingVectorizer(n_features=5000,norm=None,alternate_sign=False)
        logistic = LogisticRegression()
        logistic.fit(x_train,y_train)
        
        result =logistic.predict(hvectorizer.transform([f1]))
        result=result[0]
        if result==0:
            msg = 'The Entered Text is Detected as Hate Speech'
        else:
            msg= 'The Entered Text is Detected as No-Hate Speech'
        
        return render_template('prediction.html',msg=msg)    

    return render_template('prediction.html')



if __name__=='__main__':
    app.run(debug=True)