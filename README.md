# TutorEdge, a ML-powered tool to optimize your nascent tutoring business

## Introduction
The US tutoring/test prep industry is ~ $5 billion and rapidly growing with the rise of online tutoring. Many tutors from diverse backgrounds are starting their own online tutoring businesses on platforms such as [Wyzant](https://www.wyzant.com/). However, such sites offer minimal guidance to tutors in deciding how to set their rate and, given that rate, how much demand they might expect on the platforms.

## Overview
The goal of [TutorEdge](http://100.25.190.187:8501/) is to provide new tutors insight in how much they can charge for their services. User inputs information that would be found in a typical profile such as a short description, a more lengthy bio, their schedule availability, educational qualifications, and subjects they would like to teach. The app then generates a kernel density distribution (i.e. a smoothed histogram) for the rates of all tutors in the database, but weighted by their similarity to the input. Basically, TutorEdge returns a personalized recommendation to a user for how much tutors similar to them are charging, and does so in the form of a distribution. This way, the user gets a better sense of how much they can charge. Next, the user may then use their chosen rate as input in order to gauge their expected demand on Wyzant's platfrom: the app returns a binary response of either low demand (<1.5 hours/week) or high demand (> 1.5 hours/week).

## Approach
[tutor_cleaning](https://nbviewer.jupyter.org/github/vijayoct27/tutor_prediction/blob/master/tutor_cleaning.ipynb) I scrape and clean ~10k tutor profiles from Wyzant.
[tutor_nlp](https://nbviewer.jupyter.org/github/vijayoct27/tutor_prediction/blob/master/tutor_nlp.ipynb) I engineer features based on keywords extraction from tutor bios with tf-idf.

[tutor_rate](https://nbviewer.jupyter.org/github/vijayoct27/tutor_prediction/blob/master/tutor_rate.ipynb) I prototype the model pipelines for generating a weighted kernel density distribution of rate and in [tutor_demand](https://nbviewer.jupyter.org/github/vijayoct27/tutor_prediction/blob/master/tutor_demand.ipynb) I prototype the model pipeline for classifying tutor demand. 

[tutor_prediction](https://nbviewer.jupyter.org/github/vijayoct27/tutor_prediction/blob/master/tutor_prediction.ipynb) I build the pipeline for using these models on new test cases. 

### Front-end
The script to generate the front-end for the webapp on Streamlit is contained in `tutor_app.py`. 
Use ```streamlit run tutor_app.py``` to launch the app locally. 
