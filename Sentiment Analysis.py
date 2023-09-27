SENTIMENT ANALYSIS

import os,sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd 

df = pd.read_csv("train.csv")

df.shape

df.head()

df.info()

df.describe()

count = df.isnull().sum().sort_values(ascending = False)
percentage = ((df.isnull().sum()/len(df)*100)).sort_values(ascending=False)
missing_data = pd.concat([count,percentage],axis=1,keys=['count','percentage'])
missing_data

import matplotlib.pyplot as plt
%matplotlib inline
print(round(df.Is_Response.value_counts(normalize=True)*100,2))
round(df.Is_Response.value_counts(normalize=True)*100,2).plot(kind='bar')
plt.title('Percentage Distribution by review type')
plt.show()

df.drop(columns =['User_ID','Browser_Used','Device_Used'],inplace = True)

import re
import string
def text_clean_1(text):
    text = text.lower()
    text = re.sub('\[.*?\]','',text)
    text = re.sub('[%s]'% re.escape(string.punctuation),'',text)
    text = re.sub('\w"\d\w"','',text)
    return text
cleaned1 = lambda x: text_clean_1(x)

df['cleaned_description'] = pd.DataFrame(df.Description.apply(cleaned1))
df.head(10)

def text_clean_2(text):
    text = re.sub('[''"",,]','',text)
    text = re.sub('\n','',text)
    return text
cleaned2 = lambda x: text_clean_2(x)

df['cleaned_description_new'] = pd.DataFrame(df['cleaned_description'].apply(cleaned2))
df.head(10)

#Model training
ction import train_test_split
Independent_var = df.cleaned_description_new
Dependent_var =df.Is_Response

IV_train, IV_test,DV_train,DV_te
from sklearn.model_selection import train_test_split
Independent_var = df.cleaned_description_new
Dependent_var =df.Is_Response
​
IV_train, IV_test,DV_train,DV_test = train_test_split(Independent_var,Dependent_var,test_size = 0.1,random_state = 225)
print('IV_train :', len(IV_train))
print ('IV_test :', len(IV_test))
print('DV_train :', len(DV_train))
print('DV_test  :', len(DV_test))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

tvec = TfidfVectorizer()
clf2 = LogisticRegression(solver = "lbfgs")
from sklearn.pipeline import Pipeline

model = Pipeline([('vectorizer',tvec),('classifier',clf2)])
model.fit(IV_train,DV_train)
import sklearn.metrics
from sklearn.metrics import confusion_matrix
predictions = model.predict(IV_test)
confusion_matrix(predictions, DV_test)

#Model Predection
from sklearn.metrics import accuracy_score, precision_score, recall_score
​
print("Accuracy : ", accuracy_score(predictions,DV_test))
print("Precission : ", precision_score(predictions,DV_test,average = 'weighted'))
print("Recall : ", recall_score(predictions,DV_test,average = 'weighted'))

#Trying New Reviews
example = ["i am happy"]
result = model.predict(example)
print(result)