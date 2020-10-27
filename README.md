# Project-on-ML-Pipeline-Optimization

##Summary
#### This Project is one of the requisite projects of 'Machine Learning Engineer with Microsoft Azure Nanodegree Program'. In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model. This model is then compared to an Azure AutoML run.

##Introduction
#### In this project we were given a custom-coded model—a standard Scikit-learn Logistic Regression—the hyperparameters of which we had to optimize using HyperDrive. Later on we used AutoML to build and optimize a model on the same dataset, so that we can compare the results of the two methods.


### A. Hyperdrive Pipeline

#### We were given a starter code which had a train.py file and a jupyter file. We had to complete some sections of the code.  We had to import data from a URL, clean the data, and pass the cleaned data to the pipeline.

#### This part of the project involved the following steps:


1. First create a cluster using a new vm_size
2.The datset is craeted using TabularDatasetfactory class 
3. Split the dataset into test and train subsets
4. Create parameter sampler. Inverse regularization (C) and maximum iterations (max_iter) are the two parameters to be optimized for logistic regression model and we give a range of options to choose the best.
5.Create an estimator to run the pipeline from the train.py script
6. Create policy for early stopping: The range of options we can give in this policy like slack factor, evaluation interval can help to terminate run if the primary metric in specified parameters not found.
7. The hyperdrive_config pipeline is created using the above parameter sampler, policy and estimator
8. Submit the hyperdrive_config
9. Get the best model, accuracy, parameters and save the best model.

### B. Auto ML Pipeline


#### In this part too,w e were given a starter code.  We had to again import data from a URL, clean the data, and pass the cleaned data to the automl run.

#### This part of the project involved the following steps:
1.Data was cleaned and split to train and label
2.We had to modify the automl_config parameters and submit it
3.The best model was saved 
4. ML studio also provides the top K parameters and their individual impact on the model.


### Results

Comparing the results of the two approaches, we find that Hyperdrive optimisation gives an accuracy of 90.92% whereas AutoMl gives an accuracy of 91.79% which is almost a 1% improvement.
The power of AutoML lies in the fact that it uses very powerful ensemble techniques like XGBoost, VotingEnsemble, LigthBGM and Random Forest in its pipeline and cross validates the models by splitting data.
It is indeed a powerful and quick tool that data scientists can use to get a baseline model while working on huge datasets.



