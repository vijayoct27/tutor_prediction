# TutorEdge, a ML-powered tool to optimize your nascent tutoring business

## Table of Contents

1. [Introduction](#Introduction)
2. [Overview](#Overview)
3. [Approach](#Approach)
4. [Setup](#Setup)


### Introduction
The US tutoring industry is ~ $5 billion and rapidly growing with the rise of online tutoring. Many from diverse backgrounds are starting their own online tutoring businesses on platforms such as [Wyzant](https://www.wyzant.com/). However such sites offer minimal guidance in how to set your rate and, given that, how much demand you might expect on their platforms.

### Overview
[TutorEdge](http://100.25.190.187:8501/) provies new tutors insight in how much they can charge for their services. User inputs information found in a typical profile (short description, lengthy bio, schedule availability, educational qualifications, and subjects to teach). 
![Input page](https://github.com/vijayoct27/tutor_prediction/blob/master/notebook/homepage.png)

The app then generates a kernel density distribution (i.e. a smoothed histogram) for the rates of all tutors in the database, weighted by their similarity to the input. The user can then input their chosen rate to predict their expected demand on Wyzant's platfrom: the app returns a binary response of either low (<1.5 hours/week) or high demand (> 1.5 hours/week).

### Approach
[tutor_cleaning](https://nbviewer.jupyter.org/github/vijayoct27/tutor_prediction/blob/master/tutor_cleaning.ipynb) I scrape and clean ~10k tutor profiles from Wyzant.

[tutor_nlp](https://nbviewer.jupyter.org/github/vijayoct27/tutor_prediction/blob/master/tutor_nlp.ipynb) I engineer features based on keywords extraction from tutor bios with tf-idf.

[tutor_rate](https://nbviewer.jupyter.org/github/vijayoct27/tutor_prediction/blob/master/tutor_rate.ipynb) I prototype the model pipelines for generating a weighted kernel density distribution of rate and in [tutor_demand](https://nbviewer.jupyter.org/github/vijayoct27/tutor_prediction/blob/master/tutor_demand.ipynb) I prototype the model pipeline for classifying tutor demand. 


[tutor_prediction](https://nbviewer.jupyter.org/github/vijayoct27/tutor_prediction/blob/master/tutor_prediction.ipynb) I build the pipeline for using these models on new test cases. 

### Setup

See [Requirements](https://github.com/vijayoct27/tutor_prediction/blob/master/requirements.txt) for dependency packages.

`tutor_app.py` contains the necesssary preprocessing steps and makes use of the files in [data](https://github.com/vijayoct27/tutor_prediction/tree/master/data) generated by the IPython [notebooks](https://github.com/vijayoct27/tutor_prediction/tree/master/notebook).
The front-end can be run locally using:
```bash
streamlit run tutor_app.py
```
