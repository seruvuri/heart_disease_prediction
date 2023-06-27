import sys,os
from src.logger import logging
from src.exception import CustomException

import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.utils import save_obj

@dataclass
class DataTrasformationconfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTrasformationconfig()
    def get_data_transformation_object(self):
        try:
            numerical_columns=['Age', 
                               'BP', 'Cholesterol', 
                               'Max_HR', 'ST_depression', 
                               'Number_of_vessels_fluro']
            logging.info('numerical columns from dataset{}'.format(numerical_columns))

            categorical_columns=['Sex', 'chest_pain_type', 'FBS_over_120',
                                  'EKG_results', 'Exercise_angina',
                                    'Slope_of_ST', 'Thallium']
            logging.info('categorical columns from dataset{}'.format(categorical_columns))

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median"))
                    #("scaler",StandardScaler)
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder",OneHotEncoder())
                    #("scaler",StandardScaler(with_mean=False))
                ]
            )
            
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Reading train and test data completed")


            target_column='Heart_Disease'
            extra_column='_id'
            
            logging.info('getting preprocessor object')
            preprocessing_obj=self.get_data_transformation_object()
           
            logging.info('removing target column and excess column from train data')
            input_feature_train_df=train_df.drop([target_column,extra_column],axis=1)
            target_feture_train_df=train_df[target_column]
            
            logging.info('removing target column and excess column from test data')
            input_feature_test_df=test_df.drop(columns=[target_column,extra_column],axis=1)
            target_feature_test_df=test_df[target_column]

            
            
            logging.info('applying preprocessing object on training dataframe and testing dataframe')

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            #concatinating target feature and trained input trained data
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feture_train_df)
            ]
            
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info('saving preprocessing object')
            save_obj(self.data_transformation_config.preprocessor_obj_file_path,
                     obj=preprocessing_obj)
            
            return(
                train_arr,
                test_arr
            )
            
        except Exception as e:
            raise CustomException(e,sys)
        
  

    
