Feature Engineering
===================

Feature engineering in predictive learning refers to the process of creating or selecting relevant and informative features from raw data to improve the performance of a machine learning model. This process often requires domain knowledge and an understanding of the problem at hand. By carefully designing and engineering features, the model can capture important aspects of the data and make better predictions or classifications. Well-engineered features can significantly improve the performance of machine learning models and lead to more accurate predictions or classifications.

Tasks in Feature Engineering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Feature extraction:** Creating new features by combining or transforming existing ones. For example, extracting the month and year from a date, or calculating ratios or differences between variables.

**Feature encoding:** Representing categorical variables in a numerical form that can be understood by the model. This can involve techniques such as one-hot encoding, label encoding, or binary encoding.

**Feature scaling:** Scaling numerical features to a specific range, such as normalizing them to have zero mean and unit variance. This ensures that features with different scales do not disproportionately influence the model.

**Handling missing data:** Dealing with missing values by imputing or filling them in with appropriate values, such as the mean, median, or a more advanced technique like k-nearest neighbors imputation.

**Dimensionality reduction:** Reducing the number of features while retaining most of the relevant information. Techniques like principal component analysis (PCA) or feature selection algorithms can be used for this purpose.

**Feature interaction:** Creating new features by combining or interacting existing features to capture potential interactions or non-linear relationships between them. This can involve techniques like polynomial features or interaction terms.


Techniques in Feature Engineering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In feature engineering, correlation method and wrapper method are two approaches used for feature selection, which is the process of selecting a subset of relevant features from the available set of features.

Feature Selection
~~~~~~~~~~~~~~~~~~

* **Correlation Methods:** The correlation method involves analyzing the statistical relationship between each feature and the target variable. It measures the degree of association or dependence between two variables. In the correlation method, features with high correlation to the target variable are considered more important or informative, and thus selected for the model.

* **Wrapper Method:** The wrapper method involves evaluating the performance of a machine learning model using different subsets of features. It treats the feature selection as a search problem, where different combinations of features are evaluated to find the subset that produces the best model performance.

Synthetic data generation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Synthetic data generation in feature engineering refers to the process of creating artificial or synthetic data samples that mimic the characteristics and distribution of real-world data. Synthetic data generation can be a useful technique when the available real data is limited or lacks diversity, and it can help in overcoming issues such as data scarcity or privacy concerns.


Usage
^^^^^^

.. py:class:: easycheml.modelling.FeatureEngineering(*,pandas_dataframe,target_name)
    
    .. py:method:: feature_thru_correlation(lower_threshold, corr_method)
        
    :param pandas_dataframe: dataset in form of pandas dataframe
    :param target_name: name of the target variable in the given dataset
    :param lower_threshold: minimum correlation value
    :param corr_method: pearson, kendall, spearman

    
>>> from easycheml.modelling import FeatureEngineering 
>>> df=FeatureEngineering(pandas_dataframe='data',target_name='target_var')
>>> Relevant_Features=df.feature_thru_correlation(lower_threshold=0.4,corr_method='pearson')
>>> print(Relevant_Features)


