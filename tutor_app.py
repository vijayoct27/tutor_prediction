
# import packages
import streamlit as st 
import os



# NLP Pkgs
import pandas as pdm
import numpy as np
import matplotlib.pyplot as plt


@st.cache
def entity_analyzer(my_text):
	nlp = spacy.load('en')
	docx = nlp(my_text)
	tokens = [ token.text for token in docx]
	entities = [(entity.text,entity.label_)for entity in docx.ents]
	allData = ['"Token":{},\n"Entities":{}'.format(tokens,entities)]
	return allData


def main():
	"""Tutor Edge, made with Streamlit """

	# Title
	st.title("TutorEdge")
	st.subheader("Want to start tutoring online? Great! Use TutorEdge to understand your worth and gauge demand on the platform Wyzant.com!")

	bio = st.text_area("Enter Bio","Type Here ..")
	schedule = st.text_area("How many hours per week are you free to teach?")
	if len(bio) < 100:
		st.warning('Make sure the bio is slightly longer...')
	else:
		st.success('Success')
		# Rate
		if st.checkbox("Rate"):
			arr = np.random.normal(1, 1, size=100)
			plt.hist(arr, bins=20)
			st.pyplot()


		# Demand
		if st.checkbox("Demand"):
			st.subheader("Almost there! Just need some information:")
			rate = st.text_area("Enter Preferred Rate")

			if st.button("Show me my worth"):
				arr = np.random.normal(1, 1, size=100)
				plt.hist(arr, bins=20)
				st.pyplot()




	st.sidebar.subheader("About App")
	st.sidebar.text("A ML-powered tool to optimize your \n nascent tutoring business")

	st.sidebar.subheader("By")
	st.sidebar.text("Vijay Narayan")



if __name__ == '__main__':
	main()