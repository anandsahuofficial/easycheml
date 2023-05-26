Data Modelling
==============

Machine learning approaches are traditionally divided into three broad categories, depending on the nature of the "signal" or "feedback" available to the learning system

Approaches
^^^^^^^^^^

* Supervised learning

    * Regression
    * Classification
    
* Unsupervised learning
* Semi-supervised learning

Algorithms
^^^^^^^^^^

* Decision Trees
* Support Vector Machines
* Regressional Analysis
* Bayesian Network
* Genetic Algorithms
* Artificial Neural Network


Usage
^^^^^^

Regressors
~~~~~~~~~~

.. py:class:: easycheml.modelling.Regressors(*, dataset,target_name,train_size,val_size)
    
    .. py:method:: linear_models(select_model,tuner_parameters=None)
    .. py:method:: ensemble_models(select_model,tuner_parameters=None)
    .. py:method:: mixed_ensemble_models(select_model,estimator_models,tuner_parameters=None)
    .. py:method:: tree_models(select_model,tuner_parameters=None)
    .. py:method:: compare_ml_models(list_ensemble_models,tuner_parameters)
    .. py:method:: dnn_sequential_model(num_max_trials,num_executions_per_trial,num_epochs,num_batch_size)
        
    :param dataset: path of dataset 
    :param target_name: name of the target variable in the given dataset
    :param train_size: training ratio
    :param val_size: validation ratio

    
>>> from easycheml.modelling import Regressors 
>>> model=Regressors(feature,'LUMO_calculated',0.6,0.2)
>>> parameters = {
>>>    'n_estimators' :[50,100,200,300,400,500,600,700,800,900,1000],
>>>    'criterion' : ["squared_error", "friedman_mse", "absolute_error"],
>>>    'max_depth' : [3,5,7,9,11,13,15,17,19,21,23,25,27,29,31],
>>>    'min_samples_split' : [5,10,20,30,40,50,60,70,80,90,100],
>>>    'min_samples_leaf':[5,10,20,30,40,50,60,70,80,90,100]}

>>> # Random Forest
>>> model.ensemble_models("RF",parameters) 

>>> # Deep learning Model
>>> model.dnn_sequential_model(num_max_trials=3,num_executions_per_trial=3,num_epochs=10,num_batch_size=32)




Classifiers
~~~~~~~~~~~


.. py:class:: easycheml.modelling.Classifiers(*, dataset,target_name,train_size,val_size,additional_cols_list=None)
    
    .. py:method:: linear_models(select_model,tuner_parameters=None)
    .. py:method:: ensemble_models(select_model,tuner_parameters=None)
    .. py:method:: mixed_ensemble_models(select_model,estimator_models,tuner_parameters=None)
    .. py:method:: tree_models(select_model,tuner_parameters=None)
    .. py:method:: compare_ml_models(list_ensemble_models,tuner_parameters)
    .. py:method:: dnn_sequential_model(num_max_trials,num_executions_per_trial,num_epochs,num_batch_size)
        
    :param dataset: path of dataset 
    :param target_name: name of the target variable in the given dataset
    :param train_size: training ratio
    :param val_size: validation ratio
    :param additional_cols_list: list of columns excluded from training

    
>>> from easycheml.modelling import Classifiers 
>>> model=Classifiers(feature,'LUMO_calculated',0.6,0.2,additional_cols_list=None)

>>> model.dnn_sequential_model_opt(num_max_trials=1,num_executions_per_trial=1,num_epochs=10,num_batch_size=128)
>>> model.dnn_sequential_model(num_max_trials=3,num_executions_per_trial=3,num_epochs=10,num_batch_size=32)
>>> model.dnn_best_model(1,50,500)
>>> metrics=model.model_metrics()
>>> metrics

>>> metrics=model.ext_testing(ext_testing_data)
>>> metrics
