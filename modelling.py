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
    
    def correlation_method():
        pass
    
    def correlation_method():
        pass

    def generate_synthetic_data():
        pass
    

    pass



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


    









