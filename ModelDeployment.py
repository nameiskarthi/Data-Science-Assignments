#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pickle
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression


# In[2]:


model=pickle.load(open('logmodel.pkl','rb'))


# In[3]:


model


# In[4]:


x_train_columns=['Pclass','Age','SibSp','Parch','Fare','Sex_female','Sex_male','Embarked_C','Embarked_Q','Embarked_S']


# In[5]:


def preprocess_input(input_data):
    input_df=pd.DataFrame(input_data,index=[0])
    input_df=pd.get_dummies(input_df,columns=['Sex','Embarked'])

    for col in x_train_columns:
        if col not in input_df.columns:
            input_df[col]=0

    input_df=input_df[x_train_columns]
    return input_df


# In[6]:


st.title('Titanic Survival Prediction')
st.sidebar.header('User Input Parameters')


# In[7]:


pclass=st.sidebar.selectbox('Passenger Class',[1,2,3])
age=st.sidebar.number_input('Age',min_value=0.0,max_value=100.0,value=25.0)
sibsp=st.sidebar.number_input('Number of Siblings/Spouses Aboard',min_value=0,max_value=10,value=0)
parch=st.sidebar.number_input('Number of Parents/Children Aboard',min_value=0,max_value=10,value=0)
fare=st.sidebar.number_input('Fare',min_value=0.0,max_value=512.0,value=32.0)
sex=st.sidebar.selectbox('Sex',['Male','Female'])
embarked=st.sidebar.selectbox('Port of Embarkation',['Cherbourg','Queenstown','Southampton'])


# In[8]:


input_data={'Pclass':pclass,'Age':age,'SibSp':sibsp,'Parch':parch,'Fare':fare,'Sex':sex,'Embarked':embarked}


# In[9]:


input_df=preprocess_input(input_data)


# In[10]:


if st.sidebar.button('Predict'):
    prediction=model.predict_proba(input_df)
    survival_probability=prediction[0][1]

    st.subheader('Prediction')
    if survival_probability>0.5:
        st.write('Survived' if prediction[0][1]>0.5 else 'Not Survived')
    else:
        st.write('Survived' if prediction[0][1]>0.5 else 'Not Survived')

    st.subheader('Prediction Probability')
    st.write(prediction)


# In[ ]:




