# -*- coding: utf-8 -*-
"""
Created on Mon Nov  22 22:44:02 2021

@author: Jimit Dholakia
"""

from numpy.core.shape_base import hstack
from thefuzz import fuzz
from thefuzz import process
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import DBSCAN
import streamlit as st
import pandas as pd
from ast import literal_eval
import joblib
from sklearn.metrics import silhouette_score

st.title("Job Title Analysis")
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

########################################################################################################################
# Job Clustering
########################################################################################################################
df = pd.read_csv('Job Information.csv')


df.drop_duplicates(subset=['Job Title'], inplace=True)
# print(len(df))
df.loc[:,'Skills'] = df.loc[:,'Skills'].apply(lambda x: literal_eval(x))
df['Skills'] = df['Skills'].apply(lambda x: ', '.join(map(str, x)))
df = df[df['Skills'] != '']
# print(len(df))

data = pd.DataFrame(df['Skills'])
data.columns = ['Skills']
# data['Skills'] = data['Skills'].str[1:-1]
# data['Skills'] = [','.join(map(str, l)) for l in data['Skills']]

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(data['Skills'])

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

data['similar'] = DBSCAN(eps=0.7, min_samples=1,
                         n_jobs=-1).fit_predict(X_train_tfidf)

print('Silhouette Coefficient for Job Clustering:', silhouette_score(X_train_tfidf, data['similar']))

df2 = df.merge(data, left_index=True, right_index=True)
########################################################################################################################


########################################################################################################################
# Salary Prediction
########################################################################################################################
def load_model(model_path):
    return joblib.load(model_path)

joblib_model = load_model('salary_model_30_11.pkl')



########################################################################################################################
# Streamlit App Starts
########################################################################################################################
prob = st.sidebar.selectbox("Select Problem", [
                            "Salary Prediction", "Job Clustering"])
st.header(prob)

if prob == "Job Clustering":
    input_job = st.text_input("Enter Job Title")

    if input_job != "":
        st.write("The input Job Title is: ", input_job)
        x = process.extractBests(
            input_job, df['Job Title'], limit=4, scorer=fuzz.token_set_ratio, score_cutoff=70)
        # st.write(x)
        matched_jobs = list(map(lambda x: x[0], x))
        matched_jobs_df = pd.DataFrame.from_dict(
            data={'Job Title': list(map(lambda x: x[0], x)),
                  'Percentage Matched': list(map(lambda x: x[1], x))}
        )
        # st.table(matched_jobs_df)
        matched_groups = df2[df2['Job Title'].isin(
            matched_jobs)]['similar'].tolist()

        output_df = df2[df2['similar'].isin(matched_groups)]

        output_cols = ['Job Title', 'Skills_y']

        st.subheader('Similar Job Titles are:')
        st.table(output_df[output_cols].rename(columns={'Skills_y':'Skills'}).reset_index(drop=True))

elif prob == 'Salary Prediction':
    with st.form('salary_prediction_form'):
        title_input = st.text_input('Enter Job Title')
        skills_input = st.text_input('Enter Skills')
        
        edu_input = st.selectbox('Select Education Level', ['VOCATIONAL', 'HIGH SCHOOL', 'ASSOCIATE', 'BACHELOR', 'MASTER', 'DOCTORATE', 'NOT SPECIFIED'])

        edu_cols = ['Education_ASSOCIATE', 'Education_BACHELOR',
       'Education_DOCTORATE', 'Education_HIGH SCHOOL', 'Education_MASTER',
       'Education_NOT SPECIFIED', 'Education_VOCATIONAL']

        submitted = st.form_submit_button('Submit')

    if submitted:
        df = pd.DataFrame.from_dict(data={'Job Title': [title_input],
                'Skills': [skills_input],
                'Education': [edu_input]}, orient='columns')
        
        # st.write(df)
        st.subheader('Input Data')
        st.write('Job Title: ', title_input)
        st.write('Skills: ', skills_input)
        st.write('Education: ', edu_input)

        df = pd.DataFrame.from_dict(data={'Job Title': [title_input],
                'Skills': [skills_input],
                'Education': [edu_input]}, orient='columns')


        edu_dummies = pd.get_dummies(df['Education'], prefix='Education')
        edu_dummies = edu_dummies.reindex(columns=edu_cols, fill_value=0)
        df = pd.concat([df, edu_dummies], axis=1)
        del df['Education']

        y_pred = joblib_model.predict(df)

        st.subheader('Prediction')           
        st.write("Predicted Annual Salary: ${:,.2f}".format(y_pred[0]))



