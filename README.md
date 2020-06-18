# TutorEdge, a machine learning-powered tool to help optimize your nascent tutoring business

The US tutoring/test prep industry is ~ $5 billion and rapidly growing with the rise of online tutoring. Many tutors from diverse backgrounds are starting their own online tutoring businesses on platforms such as [Wyzant](https://www.wyzant.com/). However, such sites offer minimal guidance to tutors in deciding how to set their rate and, given that rate, how much demand they might expect on the platforms. From my own personal experience, the entire process seems rather arbitrary. Thus, I decided to tackle the problem in a data-driven way using Wyzant's database of tutor profiles. 

The primary goal of [TutorEdge](http://100.25.190.187:8501/) is to provide new tutors insight in how much they can charge for their services.
The user inputs information that would be found in a typical profile (e.g. a lengthy bio, their schedule availability, educational qualifications, subjects they will teach). TutorEdge is a webapp built in Streamlit and deployed on AWS. 
This is done by generating a kernel density distribution (i.e. a smoothed histogram) for the rates of all tutors in the database. 
