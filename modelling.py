import seaborn
import smogn
import pandas as pd

def build_ml_model():
    """
    This module helps in fitting to all the ml and dl algorithms that are available in Scikit-learn
    and other opensource packages

    Parameters
    ----------
    task_type : str, compulsory (regression, classification, clustering)

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
    pass

class feature_engineering:
    
    def feature_thru_wrapper(dataset:str,target_name:str,feat_selc_dirn:str,num_features:int,model:callable,score_param:str,cross_val:int):
        """
        Function to select features through feature selection method

        Parameter
        ---------
        dataset : training dataset
        target_name: name of the target variable
        feat_selc_dirn: SFS, SFFS, SBS,SBFS
        num_features: number_features_to_keep
        model: model to fit to select features
        score_param: neg_mean_squared_error, r2, accuracy
        cross_val: KFold(n_splits=5,shuffle=True, random_state=False))

        """
        
        from mlxtend.feature_selection import SequentialFeatureSelector as SFS

        dataset=dataset._get_numeric_data()
        features=dataset.drop([target_name], axis = 1)
        target = dataset.loc[:,target_name]

        if feat_selc_dirn=='SFS':
           forward_param=True
           floating_param=False

        elif feat_selc_dirn=='SBS':
           forward_param=False
           floating_param=False
        
        elif feat_selc_dirn=='SFFS':
           forward_param=True
           floating_param=True
        
        elif feat_selc_dirn=='SBFS':
           forward_param=False
           floating_param=True


        sfs = SFS(model, 
           k_features=num_features, 
           forward=forward_param, 
           floating=floating_param, 
           scoring=score_param,
           cv=cross_val)

        sfs = sfs.fit(features, target)

        print("\nfeature_pred_score :",sfs.k_score_)
        print("\nfeatures_name :",sfs.k_feature_idx_)
        print("\nfeatures_name :",sfs.k_feature_names_)

        Relevant_Features =dataset.loc[:, sfs.k_feature_names_]


        return Relevant_Features


    def feature_thru_correlation(dataset, target, lower_threshold, corr_method):
        """
        corr_method = pearson, kendall, spearman

        """
        dataset=dataset.reset_index()
        dataset = dataset[dataset.columns.drop((dataset.filter(regex='ndex')))]
        dataset=dataset._get_numeric_data()
        matrix = abs(dataset.corr(method=corr_method,numeric_only = True))[target].sort_values(kind="quicksort", ascending=False)
        matrix = matrix[matrix > lower_threshold]
        print("\nCorrelated Features :", matrix)

        Relevant_Features =dataset.loc[:, abs(dataset.corr(method=corr_method,numeric_only = True)[target]) > lower_threshold]
        Relevant_Features = Relevant_Features[Relevant_Features.columns.drop((Relevant_Features.filter(regex='ndex')))]
        Relevant_Features = Relevant_Features[Relevant_Features.columns.drop((Relevant_Features.filter(regex='unnamed')))]
        return Relevant_Features
        
    def generate_synthetic_data(dataset, target, k_value, samp, thres, rel, rel_type,coef):
    
        df = smogn.smoter(

            ## main arguments
            data = dataset,             ## pandas dataframe
            y = target,                 ## string ('header name')
            k = k_value,                ## positive integer (k < n)
            samp_method = samp,         ## string ('balance' or 'extreme')

            ## phi relevance arguments
            rel_thres = thres,          ## positive real number (0 < R < 1)
            rel_method = rel,           ## string ('auto' or 'manual')
            rel_xtrm_type = rel_type,   ## string ('low' or 'both' or 'high')
            rel_coef = coef             ## positive real number (0 < R)

        )
        seaborn.kdeplot(dataset[target], label = "Original")
        seaborn.kdeplot(df[target], label = "SMOGN")        
        return df


class modelling:
    
    def __init__(self, X_train, y_train,X_val,y_val,X_test,y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

    def random_forest():
        pass
        
    def gradient_boosting():
        pass

    def decision_tree():
        pass

    def dnn_sequential_model(self, X_train, y_train,X_val,y_val,X_test,y_test, dnn_log_model_dir):

        import os
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
        from tensorflow.keras.callbacks import History
        from tensorflow import keras
        from tensorflow.keras import layers
        from keras_tuner.engine.hyperparameters import HyperParameters
        from keras_tuner.tuners import RandomSearch
        from tensorflow.keras.callbacks import TensorBoard

        LOG_DIR = dnn_log_model_dir
        tensorboard = TensorBoard(log_dir=LOG_DIR)

        def build_model(hp):
            model = keras.Sequential()
            for i in range(hp.Int('num_layers', 2, 30)):
                model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                                    min_value=32,
                                                    max_value=3072,
                                                    step=32),
                                                    activation='relu'))
            model.add(layers.Dense(1, activation='linear'))
            model.compile(
                optimizer=keras.optimizers.Adam(
                    hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4,1e-4,1e-5,1e-6])),
                loss='mean_absolute_error',
                metrics=['mean_absolute_error'])
            return model


        tuner = RandomSearch(
            build_model,
            objective='val_mean_absolute_error',
        
            max_trials=15,
            executions_per_trial=3,
            overwrite=True,

            directory=os.path.normpath(r'D:\ANANDSAHU\DNN-Models'),
            project_name=LOG_DIR)


        tuner.search_space_summary()




        tuner.search(x=X_train,
                    y=y_train,
                    epochs=100,
                    batch_size=32,
                    callbacks=[tensorboard],
                    validation_data=(X_test, y_test))

        tuner.results_summary()





        pass


    









