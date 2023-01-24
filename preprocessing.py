import pandas as pd
import collections
import pandas as pd
import numpy as np


class PreProcessing:
    
    def preprocess_data(dataset,target_name,additional_cols_list=None):
        """
        This module get data from user and preprocess the data according to the form acceptable to 
        learning models
        
        Parameters
        ----------
        data : data_path, compulsory (data.csv, data.xlsx, data.dat)
        target : name_of_target_column
        features: name_of_features_columns

        algorithm : type of ml/dl model used to model data (linear_regression)
                    # regression_models:all, linear, ridge, Lasso, ElasticNet, randomforest, gradientboosting
                    # classification_models: 
                    # clustering_models:
        
        cross_validation_method : KFold, LeaveOneOut, StratifiedKFold

        ignore_warnings : bool, optional (default=True)
            When set to True, the warning related to algorigms that are not able to run are ignored.
        custom_metric : function, optional (default=None)
            When function is provided, models are evaluated based on the custom evaluation metric provided.
        prediction : bool, optional (default=False)
            When set to True, the predictions of all the models models are returned as dataframe.
        classifiers : list, optional (default="all")
            When function is provided, trains the chosen classifier(s).
        """
        print("\n#########################################")
        print("Preprocessing Data ..")
        print("#########################################\n")


        loaded_data=PreProcessing.load_data(dataset)
        print("\nShape of dataset before Preprocessing : ", loaded_data.shape)

        target = loaded_data.loc[:,target_name]
        additional_cols=loaded_data.loc[:, additional_cols_list]

        features_data=loaded_data.drop([target_name], axis = 1)
        numeric_data=PreProcessing.remove_nonnumeric_data(features_data)
        non_duplicate_data=PreProcessing.remove_duplicate_columns(numeric_data)
        non_varied_data=PreProcessing.remove_nonvariance_data(non_duplicate_data,0.1)
        non_multicollinear_data=PreProcessing.remove_multicollinearity(non_varied_data,0.9)
        cleaned_dataset = pd.concat([additional_cols, target,non_multicollinear_data], axis=1)
        print("\nShape of dataset after Preprocessing : ", cleaned_dataset.shape)
        train, validate, test=PreProcessing.train_validate_test_split(cleaned_dataset, train_percent=0.6, validate_percent=0.2,seed=None)

        return cleaned_dataset,train, validate, test

    def load_data(dataset):        
        data = pd.read_excel(dataset, index_col=0)  
        return data

    def remove_nonnumeric_data(dataset):
        """
        function to remove 
        1. non-numeric values from data
        2. duplicate columns
        3. infinity and null values
        """
        
        df1=dataset.select_dtypes(include=['float64','int64']) # taking only the Columns that contain Numerical Values    
        df2=df1.replace([np.inf, -np.inf], np.nan).dropna(axis=1) # removing infinity and null values
        return df2

    def remove_nonvariance_data(dataset,threshold_value):
        """
        function to remove non-varied data/columns

        paramter
        --------
        threshold_value: Setting variance threshold to 0 which means features that have same value in all samples.
        """        
        from sklearn.feature_selection import VarianceThreshold
        varModel =VarianceThreshold(threshold=threshold_value)
        varModel.fit(dataset)
        constArr=varModel.get_support()  #get_support() return True and False value for each feature.
        constCol=[col for col in dataset.columns if col not in dataset.columns[constArr]]
        dataset.drop(columns=constCol,axis=1,inplace=True)
        return dataset

    def remove_duplicate_columns(dataset):
        dupliCols=[]
        for i in range(0,len(dataset.columns)):
            col1=dataset.columns[i]
            for col2 in dataset.columns[i+1:]:
                if dataset[col1].equals(dataset[col2]):
                    dupliCols.append(col1+','+col2)
                                
        dCols =[col.split(',')[1] for col in dupliCols]        
        dataset = dataset.drop(columns=dCols,axis=1)
        return dataset

    def remove_multicollinearity(dataset,threshold):
        col_corr=set() # set will contains unique values.
        corr_matrix=dataset.corr() #finding the correlation between columns.
        for i in range(len(corr_matrix.columns)): #number of columns
            for j in range(i):
                if abs(corr_matrix.iloc[i,j])>threshold: #checking the correlation between columns.
                    colName=corr_matrix.columns[i] #getting the column name
                    col_corr.add(colName) #adding the correlated column name heigher than threshold value.
                        
        dataset=dataset.drop(columns=col_corr,axis=1)
        return dataset

    def train_validate_test_split(df, train_percent, validate_percent,seed):
        np.random.seed(seed)
        perm = np.random.permutation(df.index)
        m = len(df.index)
        train_end = int(train_percent * m)
        validate_end = int(validate_percent * m) + train_end
        train = df.loc[perm[:train_end]]
        validate = df.loc[perm[train_end:validate_end]]
        test = df.loc[perm[validate_end:]]
        return train, validate, test
        
    def data_scaler():
        pass

