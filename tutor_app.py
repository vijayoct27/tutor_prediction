
# import packages
import streamlit as st 
import os
import pandas as pdm
import numpy as np
import matplotlib.pyplot as plt


def main():
	"""Tutor Edge, made with Streamlit """

	# Title
	st.title("TutorEdge")
	st.subheader("So you you want to start tutoring online - that's great! TutorEdge helps you optimize your platform by giving you have insights into how much to charge for your services and how much demand you can expect on average.")
	# Enter information
	desc_input = st.text_input("Summarize your profile in one sentence")
	bio_input = st.text_area("Provide a more lengthy biographical description")
	if len(bio_input) < 1:
		st.warning('Make sure your bio is at least 200 words')
	else:
		schedule_input = st.number_input("How many hours per week are you available to teach?", min_value=1, max_value=168, step=1, value=5)
		subjects_input = st.multiselect("What subjects do you teach?", ['Algebra', 'PSAT', 'Physics', 'Computer Science'])
		edu_input = st.text_input("List all of your educational qualifications/degrees")

		st.markdown("Give me insights on:")


		# Rate
		if st.checkbox("Rate"):
			if st.button("Estimate your Rate!"):
				arr = np.random.normal(1, 1, size=100)
				plt.hist(arr, bins=20)
				st.pyplot()
				st.success('Summary of your results')


		# Demand
		if st.checkbox("Demand"):
			st.markdown('We just need your rate. If you would like more information on this, check the Rate option above.')
			rate_input = st.number_input("How much would you like to charge ($/hour)", min_value=1.0, max_value=500.0, step = 1.0, value=50.0)

			if st.button("Estimate your Demand!"):
				arr = np.random.normal(1, 1, size=100)
				plt.hist(arr, bins=20)
				st.pyplot()
				st.success('Summary of your results')




	st.sidebar.subheader("About TutorEdge")
	st.sidebar.text("A ML-powered tool to optimize your \n nascent tutoring business")

	st.sidebar.subheader("By")
	st.sidebar.text("Vijay Narayan")



if __name__ == '__main__':
	main()