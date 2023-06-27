import os,sys

import dill 
import pymongo
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

import certifi

def mongodb_con(database_name,collection_name):
    try:
        myclient=pymongo.MongoClient("mongodb+srv://sairam:sairam8662@cluster0.lyahcgb.mongodb.net/?retryWrites=true&w=majority",tlsCAFile=certifi.where())
        global collection
        logging.info('mongo db connection initiated')
        client = myclient
        db = client[database_name]
        collection = db[collection_name]
        return collection
    except Exception as e:
        raise CustomException(e,sys)
''''
def data_extraction(filename):
    try:
        global dataset
        extracted_data=collection.find()
        list_of_data=list(extracted_data)
        dataset=pd.DataFrame.from_dict(list_of_data)
        dataset.to_csv(filename,index=False,header=True)

        return dataset
    except Exception as e:
        raise CustomException(e,sys)
'''
def save_obj(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(X_train, X_test,y_train,y_test,models,hyper_patrameter):
    try:
        report={}

        #going through each model
        for i in range(len(list(models))):
            model=list(models.values())[i]
            hyper_param=hyper_patrameter[list(models.keys())[i]]
            
            gs=GridSearchCV(model,hyper_param,cv=3)
            gs.fit(X_train,y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)

            # Calcualting the r2score of train and test model
            train_model_score=accuracy_score(y_train,y_train_pred)

            test_model_score=accuracy_score(y_test,y_test_pred)
            report[list(models.keys())[i]]=test_model_score

            return report
    except Exception as e:
        raise CustomException(e,sys)
    
    #load_obj fucntion is responsible for loading the pickle file

def load_obj(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)