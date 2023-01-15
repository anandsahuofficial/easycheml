import pandas as pd
import collections
import pandas as pd
import numpy as np


class PreProcessing:
    
    def __init__(self, dataset,target,additional_cols_list=None):
        self.dataset = dataset
        self.target=target
        self.additional_cols_list=additional_cols_list
    
    def preprocess_data(self):
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

        loaded_data=self.load_data()
        target=self.target

        target = loaded_data.loc[:, self.target]
        additional_cols=loaded_data.loc[:, self.additional_cols_list]

        features_data=loaded_data.drop([self.target], axis = 1)
        numeric_data=PreProcessing.remove_nonnumeric_data(features_data)
        non_duplicate_data=PreProcessing.remove_duplicate_columns(numeric_data)
        non_varied_data=PreProcessing.remove_nonvariance_data(non_duplicate_data,0.1)
        non_multicollinear_data=PreProcessing.remove_multicollinearity(non_varied_data,0.9)
        cleaned_dataset = pd.concat([additional_cols, target,non_multicollinear_data], axis=1)
        train, validate, test=PreProcessing.train_validate_test_split(cleaned_dataset, train_percent=0.6, validate_percent=0.2,seed=None)

        return cleaned_dataset,train, validate, test

    def load_data(self):        
        data = pd.read_excel(self.dataset, index_col=0)  
        return data

    def remove_nonnumeric_data(dataset):
        """
        function to remove 
        1. non-numeric values from data
        2. duplicate columns
        3. infinity and null values
        """
        print("\nShape of dataset before Cleaning", dataset.shape)
        print("\nDatatypes in this dataset", dataset.dtypes)
        
        df1=dataset.select_dtypes(include=['float64','int64']) # taking only the Columns that contain Numerical Values    
        df2=df1.replace([np.inf, -np.inf], np.nan).dropna(axis=1) # removing infinity and null values
            
        print("\nShape of dataset after Cleaning", df2.shape)
        print("Datatypes in this dataset", df2.dtypes)
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
        print("\nConstant feature for Dataset \n \n ",constCol)    
        print("\nShape of  dataset before and after removing  Features of variance  :: ", threshold_value)
        print('\nShape before drop-->',dataset.shape)
        dataset.drop(columns=constCol,axis=1,inplace=True)
        print('\nShape after drop-->',dataset.shape)        
        return dataset

    def remove_duplicate_columns(dataset):
        dupliCols=[]
        for i in range(0,len(dataset.columns)):
            col1=dataset.columns[i]
            for col2 in dataset.columns[i+1:]:
                if dataset[col1].equals(dataset[col2]):
                    dupliCols.append(col1+','+col2)
                    
        print('\n \n \n # Total Duplicated columns in  dataset ::',len(dupliCols))
        print("\n\n # Duplicate in  dataset\n\n",dupliCols)
            
        #Get the duplicate column names for  Dataset
        dCols =[col.split(',')[1] for col in dupliCols]
        print("\n # Duplicate Columns \n \n ", dCols)
        
        #Find the count of unique columns
        print("\n # Length of Unique Columns for  dataset :: ", len(set(dCols)))        
        print('\n # Shape before droping duplicate columns for  dataset -->',dataset.shape)
        dataset = dataset.drop(columns=dCols,axis=1)
        print('\n # Shape after droping duplicate columns  dataset-->',dataset.shape)        
        return dataset

    def remove_multicollinearity(dataset,threshold):
        col_corr=set() # set will contains unique values.
        corr_matrix=dataset.corr() #finding the correlation between columns.
        for i in range(len(corr_matrix.columns)): #number of columns
            for j in range(i):
                if abs(corr_matrix.iloc[i,j])>threshold: #checking the correlation between columns.
                    colName=corr_matrix.columns[i] #getting the column name
                    col_corr.add(colName) #adding the correlated column name heigher than threshold value.
                        
        print("\n \n # Length Correlated columns for  Dataset:", len(col_corr))
        print('\n # Correlated columns for  Dataset:\n \n ',col_corr)         
        print('\n # Shape before droping Correlated duplicate columns for  dataset-->',dataset.shape)
        dataset=dataset.drop(columns=col_corr,axis=1)
        print('\n # Shape after droping Correlated duplicate columns for Original dataset-->',dataset.shape)        
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

