import numpy as np
import pandas as pd
import streamlit as st
import pickle
import time

saved_model = pickle.load(open('saved_model.pickle', 'rb'))
final_model = saved_model['model']
cv = saved_model['vectorizer']

st.set_page_config(page_title='NLP on Data Roles')
st.markdown("<h2 style='font-family:cambria; text-align: center; color:#1A224C' > NLP on Data Roles </h1>", unsafe_allow_html=True)


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

user_input = st.text_input("Type skill(s)", '')
role, scale = output(user_input.lower())




if user_input:
    if role == "error":
        st.error("Give relevant input (input didn't appear in the used datasets).")
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

images = ['ds.png','ml.png', 'da.png']
# st.image(images, width = 500, caption=[])

st.sidebar.image(images, use_column_width=True, caption=["Data Science Keywords","Machine Learning Keywords", "Data Analysis Keywords"])

st.header('About')
about = """
            This Machine Learning application was built by implementing common NLP techniques on over 30 randomly picked
            Job Descriptions for Data related roles like Data Analyst, Data Scientist and Machine Learning Engineer. 
            Detailed analysis and insights can be found on my notebook here - \n
            The datasets used can be found here - https://github.com/Abhishek-Dxt/Car-Bazaar/blob/master/used_cars_data.csv \n
            The entire project can be accessed on my GitHub - https://github.com/Abhishek-Dxt/Car-Bazaar \n
            It must be noted that the analysis is based on specific role descriptions by around 30-34 companies and is meant to get an
            estimate of role relevant skills.
            Check my other projects and contact details at - https://abhishek-dxt.github.io/
"""
st.write(about)