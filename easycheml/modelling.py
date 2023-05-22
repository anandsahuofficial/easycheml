####### DATA MANIPULATION LIBRARIES
import pandas as pd
import numpy as np
import statsmodels.api as sm
import smogn
import pickle

######### SYSTEM #############
import datetime
import sys
import copy
import os

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
from easycheml.preprocessing import PreProcessing as pre

############## tensorflow ##################
import os 
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.callbacks import History
from tensorflow import keras
from tensorflow.keras import layers
from keras_tuner.tuners import RandomSearch
from tensorflow.keras.callbacks import TensorBoard


global timestamp_var
value = datetime.datetime.fromtimestamp(datetime.datetime.now().timestamp())
timestamp_var=f"{value:%Y-%m-%d-%H-%M-%S}"


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

class FeatureEngineering:
    
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
        print("#######################################")
        print(f"FEATURE SELECTION THROUGH CORRELATION")
        print("#######################################")

        dataset=dataset.reset_index()
        dataset = dataset[dataset.columns.drop((dataset.filter(regex='ndex')))]
        dataset=dataset._get_numeric_data()
        matrix = abs(dataset.corr(method=corr_method,numeric_only = True))[target].sort_values(kind="quicksort", ascending=False)
        matrix = matrix[matrix > lower_threshold]
        
        Relevant_Features =dataset.loc[:, abs(dataset.corr(method=corr_method,numeric_only = True)[target]) > lower_threshold]
        Relevant_Features = Relevant_Features[Relevant_Features.columns.drop((Relevant_Features.filter(regex='ndex')))]
        Relevant_Features = Relevant_Features[Relevant_Features.columns.drop((Relevant_Features.filter(regex='unnamed')))]

        print("\nTarget : ", target)
        print("\nCorrelation Method : ", corr_method)
        print("\nCorrelation with Target\n", matrix)

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
    """
    Parameter

    dataset: 
    target_name: name of the target in the dataset
    train_size: splitsize of the training dataset
    val_size: splitsize of the validation dataset
    """
    
    def __init__(self, dataset,target_name,train_size:float, val_size:float):
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

        self.log_dir_path = "logfiles"
        isExist = os.path.exists(self.log_dir_path)
        if not isExist:
            os.makedirs(self.log_dir_path)
        
        self.models = "models"
        isExist = os.path.exists(self.models)
        if not isExist:
            os.makedirs(self.models)


    def linear_models(self,select_model:str,tuner_parameters=None):
        timestamp = copy.copy(timestamp_var)
        sys.stdout = Logger(f"{self.log_dir_path}/linear_model-{select_model}-logfile-{timestamp}.log")

        if select_model=='LR':        
            model=linear_model.LinearRegression()
        if select_model=='Ridge':        
            model=linear_model.Ridge()
        
    
        pass
    
    def ensemble_models(self,select_model:str,tuner_parameters=None):
        
        timestamp = copy.copy(timestamp_var)
        sys.stdout = Logger(f"{self.log_dir_path}/ensemble_model-{select_model}-logfile-{timestamp}.log")
        
        if select_model=='RF':        
            model=ensemble.RandomForestRegressor(random_state=6)
        if select_model=='GBR':        
            model=ensemble.GradientBoostingRegressor(random_state=0)
        if select_model=='hGBR':        
            model=ensemble.HistGradientBoostingRegressor(random_state=6)
        if select_model=='AdaBoost':        
            model=ensemble.AdaBoostRegressor(random_state=6)
        if select_model=='ETree':        
            model=ensemble.ExtraTreesRegressor(random_state=6)
                
        if tuner_parameters==None:
            model.fit(self.X_train, self.y_train.values)
            y_pred = model.predict(self.X_test)
        else:                
            RF_cv = RandomizedSearchCV(estimator=model,param_distributions=tuner_parameters,n_iter=30,cv=5,n_jobs=-1)
            RF_cv.fit(self.X_train,self.y_train.values.ravel())
            model=RF_cv.best_estimator_
            model.fit(self.X_train, self.y_train.values)
            y_pred = model.predict(self.X_test)

        print("\n#########################################")
        print("         ENSEMBLE MODEL SELECTED")
        print("#########################################\n")

        print("\nModel :", model)
        print("\nTuner Parameters :", tuner_parameters)            
        print(f"\nModel Metrics\n")
        
        r2_score_test=round(metrics.r2_score(self.y_test, y_pred),2)*100
        mse_test=round(metrics.mean_squared_error(self.y_test, y_pred,squared=False),3)
        mae_test=round(metrics.mean_absolute_error(self.y_test, y_pred),3)

        print('R2 score of training data : {0} %'.format(round(metrics.r2_score(self.y_train, model.predict(self.X_train)),2)*100))
        print('R2 score of Testing data : {0} %'.format(round(metrics.r2_score(self.y_test, y_pred),2)*100))
        print('RMSE of of Testing data : {0}'.format(round(metrics.mean_squared_error(self.y_test, y_pred,squared=False),3)))
        print('MAE of Testing data : {0}'.format(round(metrics.mean_absolute_error(self.y_test, y_pred),3)))
        print("#########################################\n")
        
        filename=f'{self.models}/{select_model}-{timestamp}.pickle'
        pickle.dump(model, open(filename, "wb"))

        return select_model, r2_score_test,mse_test, mae_test

    def mixed_ensemble_models(self,select_model,estimator_models,tuner_parameters):
    
        if select_model=='bagging':        
            model=ensemble.BaggingRegressor(estimator_models,random_state=6)
        
        if select_model=='voting':        
            model=ensemble.VotingRegressor(estimator_models)

        if select_model=='adaBoost_dtree':
            print('not implemented yet')
        if select_model=='stacking':
            print('not implemented yet')
                
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


    def tree_models(self,select_model:str,tuner_parameters=None):
        
        if select_model=='DTREE':        
            model=tree.DecisionTreeRegressor(random_state=6)
        
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

    def compare_ml_models(self,list_ensemble_models,tuner_parameters):

        if list_ensemble_models==None:
            list_ensemble_models=['RF','GBR','hGBR','AdaBoost','ETree']
        
        metrics = pd.DataFrame()

        for model in list_ensemble_models:
            modelname,r2_score_test, mse_test, mae_test=self.ensemble_models(model, tuner_parameters)
            temp ={
            'Model': model,
            'R2':r2_score_test,
            'MSE': mse_test,
            'MAE': mae_test
            }
            
            metrics=metrics.append(temp,ignore_index=True)            
            metrics = metrics.sort_values(['R2'], ascending=False)

        print(f"\n############ MODELS PERFORMANCE #############")
        print(metrics)
        print(f"#############################################\n")

    
    def build_model(self,hp):
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

    def dnn_sequential_model(self,num_max_trials,num_executions_per_trial,num_epochs,num_batch_size):
        """

        """       
                
        timestamp = copy.copy(timestamp_var)
        sys.stdout = Logger(f"{self.log_dir_path}/DNN-{timestamp}.log")
        LOG_DIR = f'{self.log_dir_path}/DNN-LOG-DIR-{timestamp}'
        tensorboard = TensorBoard(log_dir=LOG_DIR)    
        
        tuner = RandomSearch(
            self.build_model,
            objective='val_mean_absolute_error',        
            max_trials=num_max_trials,
            executions_per_trial=num_executions_per_trial,
            overwrite=True,
            directory=f'{self.log_dir_path}/DNN-TUNER-DIR-{timestamp}',
            project_name=LOG_DIR)

        print("\n########################")
        print(" Search for best model")    
        print("########################\n")

        tuner.search(x=self.X_train,
                    y=self.y_train,
                    epochs=num_epochs,
                    batch_size=num_batch_size,
                    callbacks=[tensorboard],
                    validation_data=(self.X_val, self.y_val))

        print("\n########################")
        print("Search Space Summary")    
        print("########################\n")
        tuner.search_space_summary()

        print("\n########################")
        print("Results Summary")    
        print("########################\n")
        
        tuner.results_summary()
        from functools import partial
        import collections

        filename=f'{self.models}/DNN-MODEL-{timestamp}.pickle'
        dictionary = collections.defaultdict(partial(collections.defaultdict, int))

        with open(filename, 'wb') as pickle_file:
            pickle.dump(dictionary, pickle_file)
        # pickle.dump(dictionary, "wb")


        # pickle.dump(tuner, open(dictionary, "wb"))
        


class Classifiers:
    """
    Parameter

    dataset: 
    target_name: name of the target in the dataset
    train_size: splitsize of the training dataset
    val_size: splitsize of the validation dataset
    """
    
    def __init__(self, dataset,target_name,train_size:float, val_size:float):
        self.dataset = dataset
        self.target = target_name
        self.val_size = val_size
        self.train_size = train_size
        train, validate, test=pre.train_validate_test_split(dataset,train_size,val_size,0)

        # train=train._get_numeric_data()
        # validate=validate._get_numeric_data()
        # test=test._get_numeric_data()

        self.X_train=train.drop([target_name], axis = 1)
        self.y_train = train.loc[:,target_name]
        self.X_val=validate.drop([target_name], axis = 1)
        self.y_val = validate.loc[:,target_name]
        self.X_test=test.drop([target_name], axis = 1)
        self.y_test = test.loc[:,target_name]

        self.num_targets=self.y_train.nunique()

        self.y_train, self.y_val,self.y_test = Classifiers.target_encoder(self.y_train,self.y_val,self.y_test)

        self.log_dir_path = "logfiles"
        isExist = os.path.exists(self.log_dir_path)
        if not isExist:
            os.makedirs(self.log_dir_path)
        
        self.models = "models"
        isExist = os.path.exists(self.models)
        if not isExist:
            os.makedirs(self.models)
    
    def target_encoder(y_train,y_val, y_test):
        from tensorflow.keras.utils import to_categorical
        le = LabelEncoder()
        le.fit(y_train)
        
        y_train_enc = le.transform(y_train)
        y_val_enc = le.transform(y_val)
        y_test_enc = le.transform(y_test)

        y_train_enc = to_categorical(y_train_enc)
        y_val_enc = to_categorical(y_val_enc)
        y_test_enc = to_categorical(y_test_enc)
        return y_train_enc,y_val_enc, y_test_enc

    def linear_models(self,select_model:str,tuner_parameters=None):
        timestamp = copy.copy(timestamp_var)
        sys.stdout = Logger(f"{self.log_dir_path}/linear_model-{select_model}-logfile-{timestamp}.log")

        if select_model=='LR':        
            model=linear_model.LinearRegression()
        if select_model=='Ridge':        
            model=linear_model.Ridge()
        
    
        pass
    
    def ensemble_models(self,select_model:str,tuner_parameters=None):
        
        timestamp = copy.copy(timestamp_var)
        sys.stdout = Logger(f"{self.log_dir_path}/ensemble_model-{select_model}-logfile-{timestamp}.log")
        
        if select_model=='RF':        
            model=ensemble.RandomForestRegressor(random_state=6)
        if select_model=='GBR':        
            model=ensemble.GradientBoostingRegressor(random_state=0)
        if select_model=='hGBR':        
            model=ensemble.HistGradientBoostingRegressor(random_state=6)
        if select_model=='AdaBoost':        
            model=ensemble.AdaBoostRegressor(random_state=6)
        if select_model=='ETree':        
            model=ensemble.ExtraTreesRegressor(random_state=6)
                
        if tuner_parameters==None:
            model.fit(self.X_train, self.y_train.values)
            y_pred = model.predict(self.X_test)
        else:                
            RF_cv = RandomizedSearchCV(estimator=model,param_distributions=tuner_parameters,n_iter=30,cv=5,n_jobs=-1)
            RF_cv.fit(self.X_train,self.y_train.values.ravel())
            model=RF_cv.best_estimator_
            model.fit(self.X_train, self.y_train.values)
            y_pred = model.predict(self.X_test)

        print("\n#########################################")
        print("         ENSEMBLE MODEL SELECTED")
        print("#########################################\n")

        print("\nModel :", model)
        print("\nTuner Parameters :", tuner_parameters)            
        print(f"\nModel Metrics\n")
        
        r2_score_test=round(metrics.r2_score(self.y_test, y_pred),2)*100
        mse_test=round(metrics.mean_squared_error(self.y_test, y_pred,squared=False),3)
        mae_test=round(metrics.mean_absolute_error(self.y_test, y_pred),3)

        print('R2 score of training data : {0} %'.format(round(metrics.r2_score(self.y_train, model.predict(self.X_train)),2)*100))
        print('R2 score of Testing data : {0} %'.format(round(metrics.r2_score(self.y_test, y_pred),2)*100))
        print('RMSE of of Testing data : {0}'.format(round(metrics.mean_squared_error(self.y_test, y_pred,squared=False),3)))
        print('MAE of Testing data : {0}'.format(round(metrics.mean_absolute_error(self.y_test, y_pred),3)))
        print("#########################################\n")
        
        filename=f'{self.models}/{select_model}-{timestamp}.pickle'
        pickle.dump(model, open(filename, "wb"))

        return select_model, r2_score_test,mse_test, mae_test

    def mixed_ensemble_models(self,select_model,estimator_models,tuner_parameters):
    
        if select_model=='bagging':        
            model=ensemble.BaggingRegressor(estimator_models,random_state=6)
        
        if select_model=='voting':        
            model=ensemble.VotingRegressor(estimator_models)

        if select_model=='adaBoost_dtree':
            print('not implemented yet')
        if select_model=='stacking':
            print('not implemented yet')
                
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


    def tree_models(self,select_model:str,tuner_parameters=None):
        
        if select_model=='DTREE':        
            model=tree.DecisionTreeRegressor(random_state=6)
        
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

    def compare_ml_models(self,list_ensemble_models,tuner_parameters):

        if list_ensemble_models==None:
            list_ensemble_models=['RF','GBR','hGBR','AdaBoost','ETree']
        
        metrics = pd.DataFrame()

        for model in list_ensemble_models:
            modelname,r2_score_test, mse_test, mae_test=self.ensemble_models(model, tuner_parameters)
            temp ={
            'Model': model,
            'R2':r2_score_test,
            'MSE': mse_test,
            'MAE': mae_test
            }
            
            metrics=metrics.append(temp,ignore_index=True)            
            metrics = metrics.sort_values(['R2'], ascending=False)

        print(f"\n############ MODELS PERFORMANCE #############")
        print(metrics)
        print(f"#############################################\n")

    def build_model(self,hp):
        model = keras.Sequential()
        for i in range(hp.Int('num_layers', 2, 30)):
            model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                                min_value=32,
                                                max_value=3072,
                                                step=32),
                                activation='relu'))
        model.add(layers.Dense(3, activation='softmax'))
        model.add(layers.Dense(self.num_targets, activation='softmax'))
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss='categorical_crossentropy',
        metrics=['accuracy'])
        return model
    
    def dnn_sequential_model(self,num_max_trials,num_executions_per_trial,num_epochs,num_batch_size):
        """

        """       
                
        timestamp = copy.copy(timestamp_var)
        sys.stdout = Logger(f"{self.log_dir_path}/DNN-{timestamp}.log")
        LOG_DIR = f'{self.log_dir_path}/DNN-LOG-DIR-{timestamp}'
        tensorboard = TensorBoard(log_dir=LOG_DIR)    
        
        tuner = RandomSearch(
            self.build_model,
            objective='val_accuracy',        
            max_trials=num_max_trials,
            executions_per_trial=num_executions_per_trial,
            overwrite=True,
            directory=f'{self.log_dir_path}/DNN-TUNER-DIR-{timestamp}',
            project_name=LOG_DIR)

        print("\n########################")
        print(" Search for best model")    
        print("########################\n")

        tuner.search(x=self.X_train,
                    y=self.y_train,
                    epochs=num_epochs,
                    batch_size=num_batch_size,
                    callbacks=[tensorboard],
                    validation_data=(self.X_val, self.y_val))

        print("\n########################")
        print("Search Space Summary")    
        print("########################\n")
        tuner.search_space_summary()

        print("\n########################")
        print("Results Summary")    
        print("########################\n")
        
        tuner.results_summary()
        from functools import partial
        import collections

        filename=f'{self.models}/DNN-MODEL-{timestamp}.pickle'
        dictionary = collections.defaultdict(partial(collections.defaultdict, int))

        with open(filename, 'wb') as pickle_file:
            pickle.dump(dictionary, pickle_file)


            
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def __getattr__(self, attr):
        return getattr(self.terminal, attr)
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)      
    def flush(self):
        self.terminal.flush()
        self.log.flush()








class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def __getattr__(self, attr):
        return getattr(self.terminal, attr)
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)      
    def flush(self):
        self.terminal.flush()
        self.log.flush()



