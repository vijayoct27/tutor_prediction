
# import packages
import streamlit as st 
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.pipeline import Pipeline

import re
import nltk
#nltk.download('stopwords')
#nltk.download('wordnet') 

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA

from stored_lists import experience_list, welcoming_list, passion_list, popular_subjects, full_subjects_list, undergrad, postgrad, certified, list_of_top_schools

def lemmatize(bio):
    
    stop_words = set(stopwords.words("english"))
    new_words = ["using", "show", "result", "large", "also", "one", "two", "new", "previously", "shown", 'math']
    stop_words = stop_words.union(new_words)

    #Remove punctuations
    text = re.sub('[^a-zA-Z]', ' ', bio)
    
    #Convert to lowercase
    text = text.lower()
    
    #remove tags
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    # remove periods
    text = text.replace('.', '').replace(',' , '')
    
    #Convert to list from string
    text = text.split()
    
    ##Stemming
    stemmer=PorterStemmer()
    #Lemmatisation
    lem = WordNetLemmatizer()
    #text = [stemmer.stem(lem.lemmatize(word)) for word in text if not word in stop_words]
    text = [lem.lemmatize(word) for word in text if not word in stop_words]
    text = " ".join(text)
    return text

def main():
	"""Tutor Edge, made with Streamlit """

	# Title
	st.title("TutorEdge")
	st.subheader("So you you want to start tutoring online - that's great! TutorEdge helps you optimize your platform by giving you have insights into how much to charge for your services and how much demand you can expect on average.")
	# Enter information

	desc_input = st.text_input("Summarize your profile in one sentence")
	desc_count = len(desc_input.split(' '))
	bio_input = st.text_area("Provide a more lengthy biographical description")
	bio_input_lemm = lemmatize(bio_input)
	bio_count = len(bio_input_lemm.split(' '))
	if bio_count < 1:
		st.warning('Make sure your bio is at least 50 words')
	else:
		schedule_input = st.number_input("How many hours per week are you available to teach?", min_value=1, max_value=168, step=1, value=5)
		subjects_input = st.multiselect("What subjects do you teach?", full_subjects_list)
		edu_input = st.text_area("List all your educational degrees and pedigree, separated by a comma.")
		number_degrees = edu_input.count(',') + 1


		number_subjects = len(subjects_input)

		num_popular_subjects = 0
		for f in subjects_input:
			if f in popular_subjects:
				num_popular_subjects += 1

		experience_count = 0
		welcoming_count = 0
		passion_count = 0
		for b in bio_input_lemm:
			if any(x in b for x in experience_list):
				experience_count += 1
			if any(x in b for x in welcoming_list):
				welcoming_count += 1
			if any(x in b for x in passion_list):
				passion_count += 1

		number_degrees = edu_input.count(',') + 1

		undergrad_count = 0
		postgrad_count = 0
		certified_count = 0
		if any(x in edu_input for x in undergrad):
			undergrad_count += 1
		if any(x in edu_input for x in postgrad):
			postgrad_count += 1
		if any(x in edu_input for x in certified):
			certified_count +=1

		top_school = 0
		if any(x in edu_input for x in list_of_top_schools):
			top_school += 1

		st.markdown("Give me insights on:")



		# Rate
		if st.checkbox("Rate"):
			rate_data = pd.read_csv('tutor_data_rate.csv').drop(columns='Unnamed: 0')
			rate_data_f = rate_data.drop(columns=['rate'])
			rate_pipe = Pipeline([
				('scaler', StandardScaler()),
				('reduce_dim', PCA(n_components=3))
				])
			rate_pipe.fit(rate_data_f)
			rate_trans = rate_pipe.transform(rate_data_f)
			rate_data_test = [number_subjects, top_school, schedule_input, bio_count, number_degrees, desc_count, num_popular_subjects]
			input_rate = rate_pipe.transform(np.array(rate_data_test).reshape(1,-1))
			weights_rate = (1. / np.linalg.norm(input_rate - rate_trans, axis=-1))

			if st.button("Estimate your Rate!"):
				rate_gkde = stats.gaussian_kde(rate_data.rate, bw_method = 0.5, weights=weights_rate)
				rate_ind = np.linspace(20, 200, 101)
				rate_kdepdf = rate_gkde.evaluate(rate_ind)
				
				plt.plot(rate_ind, rate_kdepdf, label='kde_weights', color="b")
				plt.title('Kernel Density Estimation')
				plt.ylabel('pdf')
				plt.xlabel('Rate ($/Hour)')

				st.pyplot()
				#st.success('Summary of your results')


		# Demand
		if st.checkbox("Demand"):
			st.markdown('We just need your rate. If you would like more information on this, check the Rate option above.')
			rate_input = st.number_input("How much would you like to charge ($/hour)", min_value=1.0, max_value=500.0, step = 1.0, value=50.0)
			demand_data = pd.read_csv('tutor_data_demand.csv').drop(columns='Unnamed: 0')
			demand_data_f = demand_data.drop(columns=['hours_per_week_estimate'])
			demand_pipe = Pipeline([
				('scaler', StandardScaler()),
				('reduce_dim', PCA(n_components=3))
				])
			demand_pipe.fit(demand_data_f)
			demand_trans = demand_pipe.transform(demand_data_f)
			demand_data_test = [schedule_input, bio_count, rate_input, number_subjects, welcoming_count, desc_count, num_popular_subjects, experience_count, postgrad_count]
			input_demand = demand_pipe.transform(np.array(demand_data_test).reshape(1,-1))
			weights_demand = (1. / np.linalg.norm(input_demand - demand_trans, axis=-1))

			if st.button("Estimate your Demand!"):
				demand_gkde = stats.gaussian_kde(demand_data.hours_per_week_estimate, bw_method = 0.5, weights=weights_demand)
				demand_ind = np.linspace(0, 20, 101)
				demand_kdepdf = demand_gkde.evaluate(demand_ind)

				plt.plot(demand_ind, demand_kdepdf, label='kde_weights', color="b")
				plt.title('Kernel Density Estimation')
				plt.xlabel('Hours per week estimate')
				plt.ylabel('pdf')

				st.pyplot()
				#st.success('Summary of your results')




	st.sidebar.subheader("About TutorEdge")
	st.sidebar.text("A ML-powered tool to optimize your \n nascent tutoring business")

	st.sidebar.subheader("By")
	st.sidebar.text("Vijay Narayan")

if __name__ == '__main__':
	main()