from json import dump
from flask import Flask, render_template, request, session
import pickle
import pandas as pd
from pandas.io.json import json_normalize
import csv
import os

app = Flask(__name__,template_folder = r"C:\Users\nyath\Desktop\Final Year Project")

app.secret_key = 'You Will Never Guess'                                                                                   
    
customer_churn_prediction_model = pickle.load(open('customer_churn_prediction_model.pkl','rb'))
loaded_vec = pickle.load(open("customer_churn_prediction_model.pkl", "rb"))
declarative_list = []
churning_list = []
df = pd.DataFrame

def class_scores(data_frame):

   sentiment_list = []
   for row_num in range(len(data_frame)):
      sentence = data_frame['Lapse'][row_num]

      print(sentence)

      if sentence > 0:
          sentiment_list.append("Churn")
          churning_list.append([sentence])

      if sentence <= 0:
          sentiment_list.append("Not Churn")

   data_frame['Churn Status'] = sentiment_list
   churn_clients = pd.DataFrame(churning_list, columns=['churning_clients'])
   session['churning_clients_file'] = churn_clients.to_json()

   return data_frame    

def churn(data_frame):
    churn_list = []
    for row_num in range(len(churning_list)):
        sentence = data_frame['churning_clients'][row_num]

    data_frame['churning'] = churn_list

    return data_frame

@app.route('/')
def index():
    return render_template('index_upload_and_show_data.html')


@app.route('/',methods = ['POST', 'GET'])
def uploadFile():
    if request.method == 'POST':
        uploaded_file = request.files['uploaded-file']
        df = pd.read_csv(uploaded_file)
        session['uploaded_csv_file'] = df.to_json()
        return render_template('index_upload_and_show_data_page2.html')

@app.route('/show_data')
def showData():
    # Get uploaded csv file from session as a json value
    uploaded_json = session.get('uploaded_csv_file', None)
    # Convert json to data frame
    uploaded_df = pd.DataFrame.from_dict(eval(uploaded_json))
    # Convert dataframe to html format
    uploaded_df_html = uploaded_df.to_html()
    return render_template('show_data.html', data=uploaded_df_html)
 
@app.route('/sentiment')
def Occupation():
    # Get uploaded csv file from session as a json value
    uploaded_json = session.get('uploaded_csv_file', None)
    # Convert json to data frame
    uploaded_df = pd.DataFrame.from_dict(eval(uploaded_json))
    # Apply sentiment function to get sentiment score
    uploaded_df_sentiment = class_scores(uploaded_df)
    uploaded_df_html = uploaded_df_sentiment.to_html()
    # uploaded_df_analysis = sent_polarity(uploaded_df)
    # uploaded_df_html = uploaded_df_analysis.to_html()
    return render_template('show_data2.html', data=uploaded_df_html)


@app.route('/churn_list')
def Churners():
    # Get uploaded csv file from session as a json value
    uploaded_json = session.get('churning_clients_file', None)
    # Convert json to data frame
    uploaded_df = pd.DataFrame.from_dict(eval(uploaded_json))
    # Apply sentiment function to get sentiment score
    uploaded_df_sentiment = churn(uploaded_df)
    uploaded_df_html = uploaded_df_sentiment.to_html()
    # uploaded_df_analysis = sent_polarity(uploaded_df)
    # uploaded_df_html = uploaded_df_analysis.to_html()
    return render_template('final.html', data=uploaded_df_html)



 
if __name__=='__main__':
    app.run(debug = True)
