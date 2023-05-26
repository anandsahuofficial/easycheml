==========
Tutorials
==========

Machine/Deep Learning in 3 Steps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Data Preprocessing
~~~~~~~~~~~~~~~~~~

>>> from easycheml.preprocessing import PreProcessing as p
>>> preprocessed_dataset,train, validate, test=p.preprocess_data(dataset,'target_name','list_of_specific_columns')
>>> preprocessed_dataset.to_excel("df_feature.xlsx")

Feature Engineering
~~~~~~~~~~~~~~~~~~~~

>>> from easycheml.modelling import feature_engineering as f
>>> feature=f.feature_thru_correlation('df_feature.xlsx', 'target_name', 0.4, 'pearson')



Data Modelling
~~~~~~~~~~~~~~~

>>> from easycheml.modelling import Regressors 
>>> model=Regressors('df_feature.xlsx','target_name',0.6,0.2)
>>> model.ensemble_models("RF", None)      # Random Forest Regressor 
>>> model.ensemble_models("AdaBoost",None) # AdaBoost Regressor

>>> # Hyperparameter for tuning above Random Forest Regressor
>>> parameters = {
    'n_estimators' :[50,100,200,300,400,500,600,700,800,900,1000],
    'criterion' : ["squared_error", "friedman_mse", "absolute_error"],
    'max_depth' : [3,5,7,9,11,13,15,17,19,21,23,25,27,29,31],
    'min_samples_split' : [5,10,20,30,40,50,60,70,80,90,100],
    'bootstrap':[True],
    'min_samples_leaf':[5,10,20,30,40,50,60,70,80,90,100],
                }
>>> model.ensemble_models("RF",parameters)

>>> # Deep Learning Sequential Model
>>> num_max_trials=3
>>> num_executions_per_trial=3
>>> num_epochs=10
>>> num_batch_size=32
>>> model.dnn_sequential_model(num_max_trials,num_executions_per_trial,num_epochs,num_batch_size)


.. Data Postprocessing and Visualization
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

