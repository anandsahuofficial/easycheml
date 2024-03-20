import pandas as pd
import collections
import pandas as pd
import numpy as np

class PreProcessing:

    def __init__(self,dataset,target_name,VarianceThreshold_value,multicollinear_threshold,scaling_method,additional_cols_list=None):
        self.dataset = dataset
        self.targetname = target_name
        self.additional_cols_list = additional_cols_list
        self.VarianceThreshold_value=VarianceThreshold_value
        self.multicollinear_threshold=multicollinear_threshold
        self.scaling_method=scaling_method
    
    def load_data(self):   
        data = pd.read_excel(self.dataset, index_col=0)  
        return data
    
    def remove_nonnumeric_data(self):
        """
        function to remove 
        1. non-numeric values from data
        2. duplicate columns
        3. infinity and null values
        """        
        self.features_data=self.features_data.select_dtypes(include=['float64','int64']) # taking only the Columns that contain Numerical Values    
        self.features_data=self.features_data.replace([np.inf, -np.inf], np.nan).dropna(axis=1) # removing infinity and null values

    def remove_nonvariance_data(self):
        """
        function to remove non-varied data/columns

        paramter
        --------
        threshold_value: Setting variance threshold to 0 which means features that have same value in all samples.
        """        
        from sklearn.feature_selection import VarianceThreshold
        varModel =VarianceThreshold(threshold=self.VarianceThreshold_value)
        varModel.fit(self.features_data)
        constArr=varModel.get_support()  #get_support() return True and False value for each feature.
        constCol=[col for col in self.features_data.columns if col not in self.features_data.columns[constArr]]
        self.features_data.drop(columns=constCol,axis=1,inplace=True)

    def remove_duplicate_columns(self):
        dupliCols=[]
        for i in range(0,len(self.features_data.columns)):
            col1=self.features_data.columns[i]
            for col2 in self.features_data.columns[i+1:]:
                if self.features_data[col1].equals(self.features_data[col2]):
                    dupliCols.append(col1+','+col2)
                                
        dCols =[col.split(',')[1] for col in dupliCols]        
        self.features_data = self.features_data.drop(columns=dCols,axis=1)

    def remove_multicollinearity(self):
        col_corr=set() # set will contains unique values.
        corr_matrix=self.features_data.corr() #finding the correlation between columns.
        for i in range(len(corr_matrix.columns)): #number of columns
            for j in range(i):
                if abs(corr_matrix.iloc[i,j])>self.multicollinear_threshold: #checking the correlation between columns.
                    colName=corr_matrix.columns[i] #getting the column name
                    col_corr.add(colName) #adding the correlated column name heigher than threshold value.
                        
        self.features_data=self.features_data.drop(columns=col_corr,axis=1)

    def normalize_features(self):
        """
        Normalize features in a Pandas DataFrame using specified normalization method.
        
        Parameters:
        df (DataFrame): Input DataFrame with features to be normalized.
        method (str): Normalization method to use. Options are 'min_max', 'z_score', 'robust', 'unit_vector', or '0_1'.
                    Default is 'min_max'.
        
        Returns:
        DataFrame: New DataFrame with normalized features.

        # Example usage:
        # Assuming df is your DataFrame with multiple columns
        # normalized_df = normalize_features(df, method='min_max')
        """
        # Copy the input DataFrame to avoid modifying the original
        # normalized_df = self.features_data.copy()
        
        # Iterate over each column (feature) in the DataFrame
        for column in self.features_data.columns:
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(self.features_data[column]):
                continue
            
            # Select the normalization method
            if self.scaling_method == 'min_max':
                # Min-max scaling for the current feature
                min_val = self.features_data[column].min()
                max_val = self.features_data[column].max()
                self.features_data[column] = (self.features_data[column] - min_val) / (max_val - min_val)
            elif self.scaling_method == 'z_score':
                # Z-score normalization for the current feature
                mean_val = self.features_data[column].mean()
                std_val = self.features_data[column].std()
                self.features_data[column] = (self.features_data[column] - mean_val) / std_val
            elif self.scaling_method == 'robust':
                # Robust scaling for the current feature
                median_val = self.features_data[column].median()
                q1 = self.features_data[column].quantile(0.25)
                q3 = self.features_data[column].quantile(0.75)
                iqr = q3 - q1
                self.features_data[column] = (self.features_data[column] - median_val) / iqr
            elif self.scaling_method == 'unit_vector':
                # Unit vector scaling for the current feature
                norm = (self.features_data[column] ** 2).sum() ** 0.5
                self.features_data[column] = self.features_data[column] / norm
            elif self.scaling_method == '0_1_direct':
                # Directly scale features to the range [0, 1] without division by zero
                max_val = self.features_data[column].max()
                if max_val != 0:
                    self.features_data[column] = self.features_data[column] / max_val
                else:
                    self.features_data[column] = 0  # Set all values to 0 if max_val is 0
    
    def separate_features_target(self):
        loaded_data=self.load_data()
        print("\nShape of dataset before Preprocessing : ", loaded_data.shape)

        self.target = loaded_data.loc[:,self.targetname]
        self.additional_cols=loaded_data.loc[:, self.additional_cols_list]
        self.features_data=loaded_data.drop([self.targetname], axis = 1)
    
    def join_features_target(self):
        self.cleaned_dataset = pd.concat([self.additional_cols, self.target,self.features_data], axis=1)
        print("\nShape of dataset after Preprocessing : ", self.cleaned_dataset.shape)        
        return self.cleaned_dataset

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
        print("\n#########################################")
        print("Preprocessing Data ..")
        print("#########################################\n")

        # loaded_data=self.load_data()
        # print("\nShape of dataset before Preprocessing : ", loaded_data.shape)

        # target = loaded_data.loc[:,self.targetname]
        # additional_cols=loaded_data.loc[:, self.additional_cols_list]
        # self.features_data=loaded_data.drop([self.targetname], axis = 1)
        self.separate_features_target()    
        self.remove_nonnumeric_data()
        self.remove_duplicate_columns()
        self.remove_nonvariance_data()
        self.remove_multicollinearity()
        self.normalize_features()
        # self.cleaned_dataset = pd.concat([self.additional_cols, self.target,self.features_data], axis=1)
        # print("\nShape of dataset after Preprocessing : ", self.cleaned_dataset.shape) 

        self.cleaned_dataset=self.join_features_target()       
        return self.cleaned_dataset

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
        
