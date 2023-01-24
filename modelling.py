####### DATA MANIPULATION LIBRARIES
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pathlib
import smogn
import pickle


###### DATA VISUALIZATION LIBRARIES
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
# %matplotlib inline

############ ML LIBRARIES ###########
from sklearn import impute,metrics,model_selection,linear_model,ensemble,svm,kernel_ridge,tree,experimental,neighbors
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,MinMaxScaler,StandardScaler
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from xgboost import XGBRegressor
from scipy import stats,special

############### EASYCHEML LIBRARIES ############
from preprocessing import PreProcessing as pre





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
    
    def feature_thru_wrapper(dataset:str,target_name:str,feat_selc_dirn:str,num_min_features:int,num_max_features:int,model:callable,score_param:str,cross_val:int):
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
        from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS


        dataset=dataset._get_numeric_data()
        features=dataset.drop([target_name], axis = 1)
        target = dataset.loc[:,target_name]

        if feat_selc_dirn=='EFS':
        #   fs =EFS(model, 
        #   min_features=num_min_features,
        #   max_features=num_max_features,
        #   scoring=score_param,
        #   cv=cross_val)
        # #   print("\nfeature_pred_score :",fs.best_score_*(-1))
        #   print('Selected features:', fs.best_idx_)
            pass


        else:
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
            k_features=num_max_features, 
            forward=forward_param, 
            floating=floating_param, 
            scoring=score_param,
            cv=cross_val)

            fs = sfs.fit(features, target)

            print("\nfeature_pred_score :",fs.k_score_)
            print("\nfeatures_name :",fs.k_feature_idx_)
            print("\nfeatures_name :",fs.k_feature_names_)

            Relevant_Features =dataset.loc[:, fs.k_feature_names_]

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

            data = dataset,             ## pandas dataframe
            y = target,                 ## string ('header name')
            k = k_value,                ## positive integer (k < n)
            samp_method = samp,         ## string ('balance' or 'extreme')
            rel_thres = thres,          ## positive real number (0 < R < 1)
            rel_method = rel,           ## string ('auto' or 'manual')
            rel_xtrm_type = rel_type,   ## string ('low' or 'both' or 'high')
            rel_coef = coef             ## positive real number (0 < R)

        )
        sns.kdeplot(dataset[target], label = "Original")
        sns.kdeplot(df[target], label = "SMOGN")        
        return df


class Regressors:

    
    def __init__(self, dataset,target_name,train_size, val_size):
        self.dataset = dataset
        self.target = target_name
        self.val_size = val_size
        self.train_size = train_size
        train, validate, test=pre.train_validate_test_split(dataset,train_size,val_size,0)
                
        train=train._get_numeric_data()
        validate=validate._get_numeric_data()
        test=test._get_numeric_data()

        self.X_train=train.drop([target_name], axis = 1)
        self.y_train = train.loc[:,target_name]
        self.X_val=validate.drop([target_name], axis = 1)
        self.y_val = validate.loc[:,target_name]
        self.X_test=test.drop([target_name], axis = 1)
        self.y_test = test.loc[:,target_name]

    
    def ensemble_models(self,select_model:str,tuner_parameters=None):
        
        if select_model=='RF':        
            model=ensemble.RandomForestRegressor(random_state=6)
        if select_model=='GBR':        
            model=ensemble.GradientBoostingRegressor(random_state=0)
        if select_model=='hGBR':        
            model=ensemble.HistGradientBoostingRegressor(random_state=6)
        
        if tuner_parameters==None:
            model.fit(self.X_train, self.y_train.values)
            y_pred = model.predict(self.X_test)
        else:                
            RF_cv = RandomizedSearchCV(estimator=model,param_distributions=tuner_parameters,n_iter=30,cv=5,n_jobs=-1)
            RF_cv.fit(self.X_train,self.y_train.values.ravel())
            model=RF_cv.best_estimator_
            model.fit(self.X_train, self.y_train.values)
            y_pred = model.predict(self.X_test)
            
        print(f"\n############ {select_model} MODEL METRICS #############")
        print('R2 score of training data : {0} %'.format(round(metrics.r2_score(self.y_train, model.predict(self.X_train)),2)*100))
        print('R2 score of Testing data : {0} %'.format(round(metrics.r2_score(self.y_test, y_pred),2)*100))
        print('RMSE of of Testing data : {0}'.format(round(metrics.mean_squared_error(self.y_test, y_pred,squared=False),3)))
        print('MAE of Testing data : {0}'.format(round(metrics.mean_absolute_error(self.y_test, y_pred),3)))
        print("#########################################\n")
        filename=f'{select_model}.pickle'
        pickle.dump(model, open(filename, "wb"))

    
    
    def random_forest(self,tuner_parameters=None):
        model=ensemble.RandomForestRegressor(random_state=6)

        if tuner_parameters==None:
            model.fit(self.X_train, self.y_train.values)
            y_pred = model.predict(self.X_test)
        else:                
            RF_cv = RandomizedSearchCV(estimator=model,param_distributions=tuner_parameters,n_iter=30,cv=5,n_jobs=-1)
            RF_cv.fit(self.X_train,self.y_train.values.ravel())
            model=RF_cv.best_estimator_
            model.fit(self.X_train, self.y_train.values)
            y_pred = model.predict(self.X_test)
            
        print("\n############ MODEL METRICS #############")
        print('R2 score of training data : {0} %'.format(round(metrics.r2_score(self.y_train, model.predict(self.X_train)),2)*100))
        print('R2 score of Testing data : {0} %'.format(round(metrics.r2_score(self.y_test, y_pred),2)*100))
        print('RMSE of of Testing data : {0}'.format(round(metrics.mean_squared_error(self.y_test, y_pred,squared=False),3)))
        print('MAE of Testing data : {0}'.format(round(metrics.mean_absolute_error(self.y_test, y_pred),3)))
        print("#########################################\n")
        filename='random_forest.pickle'
        pickle.dump(model, open(filename, "wb"))

    def gradient_boosting(self,tuner_parameters=None):
        model=ensemble.GradientBoostingRegressor(random_state=0)

        if tuner_parameters==None:
            model.fit(self.X_train, self.y_train.values)
            y_pred = model.predict(self.X_test)
        else:                
            RF_cv = RandomizedSearchCV(estimator=model,param_distributions=tuner_parameters,n_iter=30,cv=5,n_jobs=-1)
            RF_cv.fit(self.X_train,self.y_train.values.ravel())
            model=RF_cv.best_estimator_
            model.fit(self.X_train, self.y_train.values)
            y_pred = model.predict(self.X_test)
            
        print("\n############ Gradient Boosting Regressor #############")
        print('R2 score of training data : {0} %'.format(round(metrics.r2_score(self.y_train, model.predict(self.X_train)),2)*100))
        print('R2 score of Testing data : {0} %'.format(round(metrics.r2_score(self.y_test, y_pred),2)*100))
        print('RMSE of of Testing data : {0}'.format(round(metrics.mean_squared_error(self.y_test, y_pred,squared=False),3)))
        print('MAE of Testing data : {0}'.format(round(metrics.mean_absolute_error(self.y_test, y_pred),3)))
        print("########################################################\n")

        filename='gradient_boosting.pickle'
        pickle.dump(model, open(filename, "wb"))


        
    # def gradient_boosting():
    #     pass

    def decision_tree():
        pass

    def dnn_sequential_model(self, LOG_DIR,MODEL_DIR):
        """

        """
        import os 
        from tensorflow.keras import models
        from tensorflow.keras import layers
        from tensorflow.keras.callbacks import History
        from tensorflow import keras
        from tensorflow.keras import layers
        from keras_tuner.tuners import RandomSearch
        from tensorflow.keras.callbacks import TensorBoard

        tensorboard = TensorBoard(log_dir=LOG_DIR)

        def build_model(hp):
            model = models.Sequential()
            for i in range(hp.Int('num_layers', 2, 30)):
                model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                                    min_value=2,
                                                    max_value=20,
                                                    step=4),
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
            directory=pathlib.Path(MODEL_DIR),
            project_name=LOG_DIR)
        
        tuner.search_space_summary()

        tuner.search(x=self.X_train,
                    y=self.y_train,
                    epochs=10,
                    batch_size=16,
                    callbacks=[tensorboard],
                    validation_data=(self.X_val, self.y_val))
        tuner.results_summary()

        # dnn_models = tuner.get_best_models(num_models=5)
        # dnn_models[0].save("model.h5")
        # return dnn_models
        