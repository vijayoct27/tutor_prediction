## TutorEdge, a ML-powered tool to optimize your nascent tutoring business

### Introduction
The US tutoring/test prep industry is ~ $5 billion and rapidly growing with the rise of online tutoring. Many tutors from diverse backgrounds are starting their own online tutoring businesses on platforms such as [Wyzant](https://www.wyzant.com/). However, such sites offer minimal guidance to tutors in deciding how to set their rate and, given that rate, how much demand they might expect on the platforms.

### Overview
[TutorEdge](http://100.25.190.187:8501/) provies new tutors insight in how much they can charge for their services. User inputs information found in a typical profile (short description, lengthy bio, schedule availability, educational qualifications, and subjects to teach). The app then generates a kernel density distribution (i.e. a smoothed histogram) for the rates of all tutors in the database, weighted by their similarity to the input. The user can then input their chosen rate to predict their expected demand on Wyzant's platfrom: the app returns a binary response of either low demand (<1.5 hours/week) or high demand (> 1.5 hours/week).

### Approach
[tutor_cleaning](https://nbviewer.jupyter.org/github/vijayoct27/tutor_prediction/blob/master/tutor_cleaning.ipynb) I scrape and clean ~10k tutor profiles from Wyzant.
[tutor_nlp](https://nbviewer.jupyter.org/github/vijayoct27/tutor_prediction/blob/master/tutor_nlp.ipynb) I engineer features based on keywords extraction from tutor bios with tf-idf.

[tutor_rate](https://nbviewer.jupyter.org/github/vijayoct27/tutor_prediction/blob/master/tutor_rate.ipynb) I prototype the model pipelines for generating a weighted kernel density distribution of rate and in [tutor_demand](https://nbviewer.jupyter.org/github/vijayoct27/tutor_prediction/blob/master/tutor_demand.ipynb) I prototype the model pipeline for classifying tutor demand. 

[tutor_prediction](https://nbviewer.jupyter.org/github/vijayoct27/tutor_prediction/blob/master/tutor_prediction.ipynb) I build the pipeline for using these models on new test cases. 

### Front-end
`tutor_app.py` contains the data preprocessing and front-end response for the webapp. This can be locally run as:
```bash
streamlit run tutor_app.py
```
See [Requirements](https://github.com/vijayoct27/tutor_prediction/blob/master/requirements.txt) for necessary packages to install. 
