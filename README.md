## Table of Contents
1. [Description](#description)
2. [Acknowledgement](#acknowledgement)

This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight.
The initial dataset contains pre-labelled tweet and messages from real-life disaster. 
The aim of the project is to build a Natural Language Processing tool that categorize messages and gives the right output to understand what resources one need to help.

The Project is divided in the following Sections:

1. Data Processing, ETL Pipeline to extract data from source, clean data and save them in a proper databse structure
2. Machine Learning Pipeline to train a model able to classify text message in categories
3. Web App to show model results in real time. 

The ETL an ML Pipelines are stored in notebook files respectively ETL_Pipeline_Preparation.ipynb and ML_Pipeline_Preparation.ipynb.
Folder app contains a file run.py the programm and delievers results to master.html
Forlder data contains process_data.py that loads, cleans and saves datas.
Folder contains the file train_classifier.py which tarins the data

<a name="acknowledgement"></a>
## Acknowledgements

* [Figure Eight](https://www.figure-eight.com/) for providing messages dataset to train my model
