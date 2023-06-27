from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import os,sys

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import(
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
) 
from src.utils import evaluate_model,save_obj
from sklearn.metrics import classification_report

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('initiating model training')
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "Logistic Regressor":LogisticRegression(),
                "kNearestNeighbour":KNeighborsClassifier(),
                "Support Vector Classifier":SVC(),
                "Decission Tree":DecisionTreeClassifier(),
                "Random Forest":RandomForestClassifier(),
                "Gradient Boosting":GradientBoostingClassifier(),
                "Ada boost":AdaBoostClassifier()
            }
            hyper_params={
                "Logistic Regressor":{
                    'penalty':['l2'],
                    'solver':['liblinear'],
                    'multi_class':['ovr']
                },
                "kNearestNeighbour":{
                    'n_neighbors':[1,3,5,7,9],
                    'algorithm':['kd_tree'] 

                },
                "Support Vector Classifier":{
                    'c':[1,30,10,100],
                    'kernal':['linear','rbf']
                },
                "Decission Tree":{
                    'criterion':['entropy'],
                    'splitter':['random'],
                    'max_depth':[3],
                    'min_samples_split':[2],
                    'min_samples_leaf':[1],
                    'max_features':['auto']
                },
                "Random Forest":{
                    'n_estimators':[100],
                    'criterion':['entropy'],
                    'max_depth':[3],
                    'min_samples_split':[2],
                    'min_samples_leaf':[1],
                    'max_features':['auto']
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }

            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,
                                             y_test=y_test,models=models,hyper_patrameter=hyper_params)
            
            ##to get best model score from dict
            best_model_score=max(sorted(model_report.values()))

            ## to get best model name from dict

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            
            
            if best_model_score<0.6:
                raise CustomException("NO best mdoel found")
            logging.info("best found model on both training and testing dataset:{} {}".format(' ',best_model_name))

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)

            classification_report_list=classification_report(y_test, predicted)
            return classification_report_list
        
        except Exception as e:
            raise CustomException(e,sys)