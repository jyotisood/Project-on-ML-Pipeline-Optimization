# Project-on-ML-Pipeline-Optimization

## Summary
##### This Project is one of the requisite projects of 'Machine Learning Engineer with Microsoft Azure Nanodegree Program'. In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model. This model is then compared to an Azure AutoML run.

## Introduction
##### In this project we were given a custom-coded model—a standard Scikit-learn Logistic Regression—the hyperparameters of which we had to optimize using HyperDrive. Later on we used AutoML to build and optimize a model on the same dataset, so that we can compare the results of the two methods.

### Dataset and Algorithm:

###### The dataset is a UCI Bank Marketing dataset. The classification goal is to predict if a client will subscribe to a term deposit with the bank.

###### The dataset was given in form of a url. We had to create a tabular dataset using using TabularDatasetfactory class. The data is split in train and test sets in ratio of 77:33. It is preprocessed and cleaned by passing it through a training script train.py. 

##### Logistic Regression is used as the classification algorithm with accuracy as the primary metric.

### A. Hyperdrive Pipeline

#### We were given a starter code and a train.py script. We had to complete some sections of the code.  We had to import data from a URL, clean the data, and pass the cleaned data to the pipeline.

### Hyperparameters to be Optimized for Hyperdrive Experiment
1. RandomParameterSampling class:
Optimising hyperparameters is considered to be the trickiest part of building machine learning and artificial intelligence models. Azure Machine Learning supports the following methods:Random sampling, Grid sampling and Bayesian sampling. In random sampling, hyperparameter values are randomly selected from the defined search space vs grid search which takes into account all sets of  all possible values. Random sampling has proven to provide similar results to grid search but is more faster and uses computing resources wisely. Hence, for this experiment, I used random parameter sampling.

In this Parameter values of the parameter space are chosen from a set of discrete values or distribution over a continuous range. I used the choice function to define our 2 parameters- inverse regularization(C) and maximum iterations(max_iter), with a choice of set of discrete values of parameters for C in (100, 10, 1.0, 0.1, 0.01), and for max_iter in (25, 50, 100,125).

#### The benefits of using these parameters are:
 a. Inverse regularization parameter(C)- Regularization is applying a penalty to increasing the magnitude of parameter values in order to de-incentivize overfitting. Here we used an inverse regularisation parameter C =1/λ. Lowering C - would strengthen the Lambda regulator.
 
 b. Max_iter: Maximum number of iterations taken for the solvers to converge. We can vary it according to the number of times we want our experiment to learn from data.

2. Policy for Early Stopping: 
We used Bandit policy for the early stopping of the experiment. This policy uses criteria like slack factor, slack amount, and evaluation_interval as its parameters. Any run that doesn't fall within the slack factor or slack amount with respect to the primary metric gets terminated and saves time and compute resources.


#### This part of the project involved the following steps:

1. First create a cluster using a new vm_size
2. The datset is craeted using TabularDatasetfactory class 
3. Split the dataset into test and train subsets
4. Create parameter sampler. Inverse regularization (C) and maximum iterations (max_iter) are the two parameters to be optimized for logistic regression model and we give a range of options to choose the best.
5. Create an estimator to run the pipeline from the train.py script
6. Create policy for early stopping: The range of options we can give in this policy like slack factor, evaluation interval can help to terminate run if the primary metric in specified parameters not found.
7. The hyperdrive_config pipeline is created using the above parameter sampler, policy and estimator
8. Submit the hyperdrive_config
9. Get the best model, accuracy, parameters and save the best model.


### B. Auto ML Pipeline

###### In this part too,we were given a starter code.  We had to again import data from a URL, split the data and pass it to the automl run.

#### This part of the project involved the following steps:
1. Data was cleaned and split to train and label
2. We had to modify the automl_config parameters and submit it
3. The best model was saved 
4. AutoML also provides a glimpse of the top K parameters (can be seen in the studio) and their individual impact on the model.

######  Voting Ensemble was the best algorithm in the automl run with an accuracy of 91.31%. The top K features that impact the model performance included Duration, nr.employed, emp.var.rate.
######  The parameters provided by the best run indicate that model used datatransformer preprocessing, used 25 estimators, and weights in the range [0.13333333333333333, 0.2, 0.06666666666666667,  0.2, 0.06666666666666667,  0.06666666666666667, 0.06666666666666667, 0.13333333333333333, 0.06666666666666667]




### Results

Comparing the results of the two approaches, we find that Hyperdrive optimization gives an accuracy of 90.92% whereas AutoMl gives an accuracy of 91.31% which is almost a 0.40% improvement.
The power of AutoML lies in the fact that it uses very powerful ensemble techniques like XGBoost, VotingEnsemble, LigthBGM and Random Forest in its pipeline and cross validates the models by splitting data.
It is indeed a powerful and quick tool that data scientists can use to get a baseline model while working on huge datasets.


### Future Improvements and Recommendations

In future, the hyperparameters can further be tuned up in hyperdrive for getting better results. Experiment can also be iterated over more number of times to see if performance of the model improves over more iterations.




