Data Preprocessing
====================

The efficiency of any predictive model depends on the quality of its training data. Data's quality is determined by its accuracy (whether it is correct or incorrect, accurate or not), completeness (whether it is recorded or not), consistency (whether it has been modified or not), timeliness (whether it has been updated recently), believability (how likely it is that the data are accurate), and interpretability (how simple it is to understand the data). 

Tasks in Data Preprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Data cleaning:** Fill in missing values, smooth noisy data, identify or remove outliers, and resolve inconsistencies.

**Data integration:** Integration of multiple databases, data cubes, or files.

**Data reduction:** Dimensionality reduction, Numerosity reduction, Data compression

**Data transformation and discretization:** Normalization, Concept hierarchy generation

Usage
^^^^^

.. py:class:: easycheml.preprocessing(*,dataset,target_name,VarianceThreshold_value,multicollinear_threshold additional_cols_list=None)

    .. py:method:: preprocess_data()

    :param dataset: path of the dataset 
    :param target_name: name of the target variable in the given dataset
    :param VarianceThreshold_value: feature selector that removes all the low variance features from the dataset that are of no great use in modeling.
        
        * If Variance Threshold = 0 (Remove Constant Features )
        * If Variance Threshold > 0 (Remove Quasi-Constant Features )
    
    :param multicollinear_threshold: Multicollinearity is a situation where two or more predictors are highly linearly related
        
        * 0 < Multicollinear Threshold < 1 

    :param additional_cols_list: list of columns excluded in preprocessing

    
>>> from easycheml.preprocessing import PreProcessing 
>>> df= PreProcessing(dataset='datapath',target_name='target_var',VarianceThreshold_value=0.1,multicollinear_threshold=0.9,additional_cols_list=None)
>>> preprocessed_dataset=df.preprocess_data()
>>> preprocessed_dataset.to_excel("df_preprocessed_data.xlsx")


