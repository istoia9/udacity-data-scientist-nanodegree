# Disaster Response Pipeline Project

## Project

A Machine Learning pipeline was coded in this project. 

The pipeline's goal is to analyse some Figure Eight's data, respectively messages sent by people who found themselves in disaster situations, and train a ML model which will tag a new message accordingly.

The ML model is a sklearn pipeline object which uses a MultiOutputClassifier(KNeighborsClassifier()) as estimator. GridSearch is applied in order to enable parameters' selection.

The process' stages are:
a. ETL - extracts and transforms the raw data, and loads them to a SQLAlchemy database
b. Machine Learning - fits a model to the data previously saved in the database, predicting message tags
c. Dashboard - published HTML which brings some visualizations on the database itself, besides message classifications

## File Structure

	- app
	| - template
	| |- master.html
	| |- go.html
	|- run.py

	- data
	|- disaster_categories.csv
	|- disaster_messages.csv
    |- DisasterResponse.db
	|- process_data.py

	- models
	|- train_classifier.py
	|- classifier.pkl

	- README.md

## Instructions given by Udacity in order to run the code properly:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/