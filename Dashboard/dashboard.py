# -*- coding: utf-8 -*-
"""
Created on Tue May 24 02:16:06 2022

@author: 60112
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt

#from nltk.tokenize import word_tokenize
#from collections import Counter
#from nltk import ngrams
import seaborn as sns
import pandas as pd
import streamlit as st


#%matplotlib inline
import re
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import seaborn as sns
import joblib,pickle


def draw_pie_chart():
    #read the data
    data = pd.read_csv('sentiment1.csv')
    
    #count each sentiment
    sent = data['sentiment'].value_counts()
    
    #store the sentiment
    store = []
    for i in sent:
        store.append(i)
        
    #labels for annote
    mylabels = ["Good", "Not Bad", "Very good", "Not Good", "Bad", "Very Bad"]
    

   #visualize pie chart to show the trend

   # Creating explode data
    explode = (0.1, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Creating color parameters
    colors = ( "orange", "cyan", "#E3CF57",
              "grey", "#EE6A50", "beige")

    # Wedge properties
    wp = { 'linewidth' : 1 }

    # Creating autocpt arguments
    def func(pct, allvalues):
        absolute = int(pct / 100.*np.sum(allvalues))
        return "{:.1f}%\n({:d})".format(pct, absolute)

    # Creating plot
    fig, ax = plt.subplots(figsize =(10,7 ))
    wedges, texts, autotexts = ax.pie(store,
                                      autopct = lambda pct: func(pct, store),
                                      explode = explode,
                                      labels = mylabels,
                                      shadow = True,
                                      colors = colors,
                                      startangle = 90,
                                      wedgeprops = wp,
                                      textprops = dict(color ="black"))

    # Adding legend
    ax.legend(wedges, mylabels,
              title ="Sentiments",
              loc ="center left",
              bbox_to_anchor =(1.2, 0, 0.5, 1.5))

    plt.setp(autotexts, size = 8, weight ="bold")
    #ax.set_title("Each Sentiments by Percentage and Total number")

    # show plot
    plt.show()
    
    return fig
    

#draw_pie_chart()



#draw word frequency
def word_fequncy_all():
    df = pd.read_csv('Word_freq.csv')
    
    fig= plt.figure(figsize=(12,10))
    
    sns.barplot(x='Frequency', y='Word',data=df[4:24])
   # plt.ylim([0,30])
    plt.xticks(rotation = 15, fontsize = 10)
    plt.yticks(fontsize = 15)
    plt.xlabel('Frequency', fontsize = 20)
    plt.ylabel('Words', fontsize = 20)
    plt.title('Top 20 Most frequent Words', fontsize = 20)
    plt.show()
    
    return fig
    
word_fequncy_all()

#draw word frequency by sentiment
def word_fequncy_each_sentiment(choice):
    
    if choice == 'Very Good':
        df = pd.read_csv('verygood_freq.csv')
        df1=pd.read_csv('sent_veryGood.csv')
        review = pd.DataFrame(df1['review'])
    elif choice == 'Good':
        df= pd.read_csv('good_freq.csv')
        df1=pd.read_csv('sent_Good.csv')
        review = pd.DataFrame(df1['review'])
    elif choice == 'Not Bad':
        df= pd.read_csv('notbad_freq.csv')
        df1=pd.read_csv('sent_notbad.csv')
        review = pd.DataFrame(df1['review'])
    elif choice == 'Not Good':
        df= pd.read_csv('notgood_freq.csv')
        df1=pd.read_csv('sent_notgood.csv')
        review = pd.DataFrame(df1['review'])
    elif choice == 'Bad':
        df = pd.read_csv('bad_freq.csv')
        df1=pd.read_csv('sent_bad.csv')
        review = pd.DataFrame(df1['review'])
    elif choice =='Very Bad':
        df = pd.read_csv('verybad_freq.csv')
        df1=pd.read_csv('sent_verybad.csv')
        review = pd.DataFrame(df1['review'])
    else: 
        print("please select your choice")
    
   
    
    fig= plt.figure(figsize=(12,10))
    
    sns.barplot(x='Frequency', y='Word',data=df[4:24])
   # plt.ylim([0,30])
    plt.xticks(rotation = 15, fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.xlabel('Frequency', fontsize = 20)
    plt.ylabel('Words', fontsize = 20)
    #plt.title('Top 30 Most frequent Words in'+" "+"'"+choice+"'" + " "+ "Sentiment", fontsize = 20)
    plt.show()
    
    return fig,review



#please place the stopwordlist.txt into the same folder where you run this code
f=open('stopwordlist.txt',encoding='utf-8')
stop_words=[a.replace('\n','') for a in f.readlines()]

def cleanise(txt):
  #sentence = Sentence(txt)
  #txt = str(spellChecker.spellCheck(sentence))
  txt=txt.replace("İ","i");
  txt=txt.replace("I","ı");
  txt=txt.lower()
  txt = re.sub(r'[^\w\s]', ' ', txt)
  txt = ''.join(ch for ch in txt if (ch.isalnum() or ch==' '))
  txt=txt.replace("  "," ")
  txt=txt.replace("\t"," ")
  txt=txt.replace("\n","")
  newtxt=""
  for s in txt.split(" "):
    if s not in stop_words:
      newtxt=newtxt+s+" "
  newtxt=newtxt.strip()
  return newtxt

#def prediction(text):
 #   text= [cleanise(text)]
  #  pred =classifier.predict(text)
   # return pred


from streamlit_option_menu import option_menu
#config on basic settings of stramlit
st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center; color: grey;marginTop: -85px'>Sentiment Analysis</h1>", unsafe_allow_html=True)
#st.title("Sentiment Analysis on Turkish Movie")
#option1 = st.sidebar.selectbox('App Navigation?', ('Overview','Analysis by each score','Real time opinion analysis'))


pages = {
  "main": "Real time opinion analysis",
  "page2":"Analysis by each score from Dataset",
  "page3": "Overview"
}

#with st.sidebar:
 #   selected = option_menu("Menu", ["page 1", 'page 2', 'page 3'], 
  #       default_index=1)

option = st.sidebar.radio("Select page", pages.values())



if(option == pages['page3']):
#if selected == "page 2":
    
    ### using columns to make the visualization looks better
    col = st.columns(2)
    a = draw_pie_chart()
    b = word_fequncy_all()
    with col[0]:
        st.markdown("<h3 style= 'color: grey;'>Sentiment Level Analysis</h3>", unsafe_allow_html=True)
        st.markdown("<h6 style= 'color: grey;'>Overall each sentiment by percentage and total number</h6>", unsafe_allow_html=True)
        st.write(a)
    with col[1]:
        st.markdown("<h3 style= 'color: grey;'>Word Frequency Level Analysis</h3>", unsafe_allow_html=True)
        st.markdown("<h6 style= 'color: grey;'>Word Frequency</h6>", unsafe_allow_html=True)
        st.write(b)
   #st.write(followers_story_str)
   
elif(option == pages['page2']):
#elif selected == "page 3":
    option1 = st.sidebar.selectbox('Visualization by Score?', ("",'5','4','3','2','1','0'))
    col2 = st.columns(2)   
    if(option1 == '5'):
        choice = "Very Good"
        f,r = word_fequncy_each_sentiment(choice)
        
        with col2[0]:
            st.markdown("<h4 style= 'color: grey;'>Word Frequency Level Analysis by each Score</h4>", unsafe_allow_html=True)
            st.markdown("<h6 style= 'color: grey;'>Top 20 Most frequent Words in "+choice+" Sentiment</h6>", unsafe_allow_html=True)
            st.write(f)
        
        with col2[1]:
           st.markdown("<h4 style= 'color: grey;'>Associated Comments</h4>", unsafe_allow_html=True)
           st.markdown("<h6 style= 'color: grey;'>Reviews from " +choice+ " Sentiment</h6>", unsafe_allow_html=True)
           st.write(r)
           
    elif(option1 == '4'):
        choice = "Good"
        f,r = word_fequncy_each_sentiment(choice)
        
        with col2[0]:
            st.markdown("<h4 style= 'color: grey;'>Word Frequency Level Analysis by each Score</h4>", unsafe_allow_html=True)
            st.markdown("<h6 style= 'color: grey;'>Top 30 Most frequent Words in "+choice+" Sentiment</h6>", unsafe_allow_html=True)
            st.write(f)
        
        with col2[1]:
           st.markdown("<h4 style= 'color: grey;'>Associated Comments</h4>", unsafe_allow_html=True)
           st.markdown("<h6 style= 'color: grey;'>Reviews from " +choice+ " Sentiment</h6>", unsafe_allow_html=True)
           st.write(r)
           
    elif(option1 == '3'):
        choice = "Not Bad"
        f,r = word_fequncy_each_sentiment(choice)
        
        with col2[0]:
            st.markdown("<h4 style= 'color: grey;'>Word Frequency Level Analysis by each Score</h4>", unsafe_allow_html=True)
            st.markdown("<h6 style= 'color: grey;'>Top 30 Most frequent Words in "+choice+" Sentiment</h6>", unsafe_allow_html=True)
            st.write(f)
        
        with col2[1]:
           st.markdown("<h4 style= 'color: grey;'>Associated Comments</h4>", unsafe_allow_html=True)
           st.markdown("<h6 style= 'color: grey;'>Reviews from " +choice+ " Sentiment</h6>", unsafe_allow_html=True)
           st.write(r)
           
    elif(option1 == '2'):
        choice = "Not Good"
        f,r = word_fequncy_each_sentiment(choice)
        
        with col2[0]:
            st.markdown("<h4 style= 'color: grey;'>Word Frequency Level Analysis by each Score</h4>", unsafe_allow_html=True)
            st.markdown("<h6 style= 'color: grey;'>Top 30 Most frequent Words in "+choice+" Sentiment</h6>", unsafe_allow_html=True)
            st.write(f)
        
        with col2[1]:
           st.markdown("<h4 style= 'color: grey;'>Associated Comments</h4>", unsafe_allow_html=True)
           st.markdown("<h6 style= 'color: grey;'>Reviews from " +choice+ " Sentiment</h6>", unsafe_allow_html=True)
           st.write(r)
           
    elif(option1 == '1'):
        choice = "Bad"
        f,r = word_fequncy_each_sentiment(choice)
        
        with col2[0]:
            st.markdown("<h4 style= 'color: grey;'>Word Frequency Level Analysis by each Score</h4>", unsafe_allow_html=True)
            st.markdown("<h6 style= 'color: grey;'>Top 30 Most frequent Words in "+choice+" Sentiment</h6>", unsafe_allow_html=True)
            st.write(f)
        
        with col2[1]:
           st.markdown("<h4 style= 'color: grey;'>Associated Comments</h4>", unsafe_allow_html=True)
           st.markdown("<h6 style= 'color: grey;'>Reviews from " +choice+ " Sentiment</h6>", unsafe_allow_html=True)
           st.write(r)
           
    elif(option1 == '0'):
        choice = "Very Bad"
        f,r = word_fequncy_each_sentiment(choice)
        
        with col2[0]:
            st.markdown("<h4 style= 'color: grey;'>Word Frequency Level Analysis by each Score</h4>", unsafe_allow_html=True)
            st.markdown("<h6 style= 'color: grey;'>Top 30 Most frequent Words in "+choice+" Sentiment</h6>", unsafe_allow_html=True)
            st.write(f)
        
        with col2[1]:
           st.markdown("<h4 style= 'color: grey;'>Associated Comments</h4>", unsafe_allow_html=True)
           st.markdown("<h6 style= 'color: grey;'>Reviews from " +choice+ " Sentiment</h6>", unsafe_allow_html=True)
           st.write(r)

elif(option == pages['main']):
#elif selected == "page 1":
    #classifier = joblib.load('tf-idf_svm.pkl')
    # display the front end aspect
    #st.markdown(html_temp, unsafe_allow_html = True) 
    category = ["VeryBad","Bad","NotGood","NotBad","Good","VeryGood"]
    # following lines create boxes in which user can enter data required to make prediction
    st.markdown("<h4 style= 'color: grey;'>Write your Opinion here</h4>", unsafe_allow_html=True)
    text = st.text_input(" ")
    result =""
    text1 = cleanise(text)
    s1 = [text1]
    # when 'Predict' is clicked, make the prediction and store it 
    button = st.button("Predict by ML model")
    
    if button:
        Dict = {}
        for cat in category:
            model = pickle.load(open("model_svm_tfidf_parag_"+ cat +".pkl",'rb'))
            result = model.predict(s1)
            Dict[cat] = result
            
        sent = []
        for key in list(Dict.keys()):  # Use a list instead of a view
            if Dict[key] == 1:
                sent.append(key)
               
                
        if (len(sent) == 0):
            #    res=prediction(text)
                #result = .append(res)
            res = "Predicted sentiment is Neutral"
           # st.session_state.res1 = res
            #st.success(format(res))
            st.markdown(f'<p style="background-color:#FFFF00;color:black;font-size:24px;border-radius:2%;">{res}</p>', unsafe_allow_html=True)
            
        else:
            if(sent[0] =='VeryGood' or sent[0]=='Good'):
            #st.session_state.cs = sent
            #st.success('Predicted Sentiment is {}'.format(sent))
            #st.markdown("<h4 style= 'color: grey;'>Predicted Sentiment is {}'.format(sent)</h4>", unsafe_allow_html=True)
                st.markdown(f'<p style="background-color:#00FF00;color:black;font-size:24px;border-radius:2%;">{"Predicted sentiment is Positive"}</p>', unsafe_allow_html=True)         
            elif(sent[0] =='NotGood'or  sent[0]=='NotBad'):
                st.markdown(f'<p style="background-color:#FFFF00;color:black;font-size:24px;border-radius:2%;">{"Predicted sentiment is Neutral"}</p>', unsafe_allow_html=True)
            elif(sent[0] =='VeryBad' or sent[0] =='Bad'):
                st.markdown(f'<p style="background-color:#FF4D38;color:black;font-size:24px;border-radius:2%;">{"Predicted sentiment is Negative"}</p>', unsafe_allow_html=True)
    
                



























