# Udacity Project: Pipeline to Classify Disaster Response Messages



### Instructions to run the scripts and the web app:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


<a id='installation'></a>
### Installation:

This project was developed using python 3.7. A file with the package dependencies of the conda environment is included in the project.

<a id='project_motivation'></a>
### Project Motivation:

This project was developed in the Udacity Data Scientist Nanodegree. Its motivation is, given a csv file with messages related with different kind of dissasters and a previous classification of these messages, to develop a `Machine Learning Pipeline` to classify any other message.

The result should be implemented as a web app that will show us some graphs with statistics or distributions of the messaages and information about the accuracy of the model used in the web application.


<a id='file_descriptions'></a>
### File Descriptions


    disaster_response_pipeline_project
    
    │   disaster_response.yml             : export of the conda environment.
    │   README.md                         : README.md file
    │
    ├───app
    │   │   run.py                        : script to initialize Flask, implement the routes and run the web application
    │   │
    │   └───templates
    │           go.html                   : html template to visualize the classification model response.
    │           master.html               : html template to display the main pate with inforation about the model and the 
    │                                       dataset,
    ├───data
    │       DisasterResponse.db           : SQLite database result of prcossing the csv data with the process_data.py script. 
    │       disaster_categories.csv       : categories assigned previously to each message used to train the model.
    │       disaster_messages.csv         : list of messages used to train the model.
    │       process_data.py               : script to precess the csv files and save the result in a SQLite table.
    │
    ├───models
    │       classifier.pkl                : object of the model trained with the dataset given.
    │       results.csv                   : measures of precision, recall, f1-score, etc, obtained after training the model.
    │       train_classifier.py           : script to train the machin learning model
    │
    ├───Notebook Docs
    │       ETL Pipeline Preparation.ipynb : Notebook with the previous ETL works to build the process_data.py script.
    │       ML Pipeline Preparation.ipynb  : Work with the Machine Learning analysis to build the train_classifier.py en run.py
    │                                        scripts.
    │
    └───transformers
            my_transformers.py          : script with a list of custom transformes to be used in run.py and
                                          train_classifier.py.


<a id='results'></a>
### Results

We have started training two Pipelines applying a MultioutputClassifier to RandomForest and LinearSVC models. The features used to train the modes where the TFIDF features extracted from the messages.

Then we tried training other models like KNeighborsClassifier, MultinomialNB and SGDClassifier, but they didn't improve the performance of the first two.

We tried to apply Grid Search to tune some hyperparameters in the RandomForest and LinearSVC models.

And finally we have designed other Pipelines adding custom transformers like searching for Organizacions, Persons, GPE anotations into the messages, etc. We have also considered to add other features like the length of the message and the Genre (origin of the message: Media, Direct or News).

In general we haven't acquaire a great improvement, but we have seen that normally Random Foreset is the model with best precision. This value is even better if we add the additional features we have mentioned besides the TFIDF.

On the other hand LinearSVC has showed the best results in terms of f1-score. We haven't seen important varitations adding the new features to the model.

