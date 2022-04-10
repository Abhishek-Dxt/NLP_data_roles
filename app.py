import numpy as np
import pandas as pd
import streamlit as st
import pickle
import time

saved_model = pickle.load(open('saved_model.pickle', 'rb'))
final_model = saved_model['model']
cv = saved_model['vectorizer']

st.set_page_config(page_title='NLP on Data Roles')
st.markdown("<h2 style='font-family:cambria; text-align: center; color:#1A224C' > NLP on Data Roles </h2>", unsafe_allow_html=True)


def output(skills):
    data = cv.transform([skills]).toarray()
    probas = final_model.predict_proba(data)
    # print(probas[0][0])
    if probas[0][0] == 0.3074074074074075:
        return "error", [0,0,0]
    classes = final_model.classes_
    lst = []
    role = final_model.predict(data)[0]
    for class_name, proba in zip(classes, probas[0]):
        lst.append(round(proba * 100, 2))
    return role, lst

user_input = st.text_input("Enter skill(s) e.g. TensorFlow, MapReduce, Tableau, or try combinations.", '')
role, scale = output(user_input.lower())




if user_input:
    if role == "error":
        st.error("Give more relevant input.")
    else:
        st.write('Data Analyst - ', str(scale[0]), '%')
        my_bar_da = st.progress(0)
        for percent_complete in range(int(scale[0])):
            time.sleep(0.01)
            my_bar_da.progress(percent_complete + 1)

        st.write('Data Scientist - ', str(scale[1]), '%')
        my_bar_ds = st.progress(0)
        for percent_complete in range(int(scale[1])):
            time.sleep(0.01)
            my_bar_ds.progress(percent_complete + 1)

        st.write('Machine Learning Engineer - ', str(scale[2]), '%')
        my_bar_ml = st.progress(0)
        for percent_complete in range(int(scale[2])):
            time.sleep(0.01)
            my_bar_ml.progress(percent_complete + 1)

        st.success("Ideal Role: "+role)

with st.sidebar:
    st.markdown("<h3 style='font-family:cambria; text-align: center; color:#8b0000' > Dataset analysis and Visualization </h3>", unsafe_allow_html=True)

images = ['ds.png','ml.png', 'da.png']
st.sidebar.image(images, use_column_width=True, caption=["Data Science Keywords","Machine Learning Keywords", "Data Analysis Keywords"])

st.header('About')
about = """
            This Machine Learning application was built by implementing common NLP techniques on over 30 randomly picked
            Role Descriptions for data related titles like Data Analyst, Data Scientist and Machine Learning Engineer. 
            Detailed analysis and insights can be found on my notebook here - https://github.com/Abhishek-Dxt/NLP_data_roles/blob/master/NLP_data_roles.ipynb \n
            It must be noted that the analysis is based on specific role descriptions by around 30-34 companies and is meant to get an
            estimate of role relevant skills.\n
            The entire project along with the datasets used can be found in my repository here - 
            https://github.com/Abhishek-Dxt/NLP_data_roles \n
            Check my other projects and contact details at - https://abhishek-dxt.github.io/
"""
st.write(about)
