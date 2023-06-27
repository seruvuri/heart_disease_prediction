import os,sys
import pandas as pd
import pymongo
from src.logger import logging
from src.exception import CustomException
from src.utils import mongodb_con
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer



@dataclass
class dataingestionconfig:
    raw_data_file_path=os.path.join('artifacts','raw.csv')
    test_data_file_path=os.path.join('artifacts','test.csv')
    train_data_file_path=os.path.join('artifacts','train.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=dataingestionconfig()

    def initiate_data_ingestion(self):
        try:
            global dataset_df
            logging.info('data ingestion started')

            logging.info('Retriving MongoDB client details')
            collection=mongodb_con(database_name='mydatabase',collection_name='heartdisease')
            
            logging.info('Data is extracting from MongoDB')
            #reading data from mongoDB as dastaframe
            dataset_df=pd.DataFrame(list(collection.find()))
            dataset_df.to_csv(self.ingestion_config.raw_data_file_path,index=False,header=True)
            logging.info('Extracted data is available in {file_name}'.format(file_name=self.ingestion_config.raw_data_file_path))

            #print(dataset_df.dtypes)
            
            logging.info('splitting dataset into test and train initiated')
            #splitting data into test and train
            train_set,test_set=train_test_split(dataset_df,test_size=0.3,random_state=42)
        
            #saving train data to artifacts directory 
            train_set.to_csv(self.ingestion_config.train_data_file_path,index=False,header=True)

            #saving test data to artifacts directory
            test_set.to_csv(self.ingestion_config.test_data_file_path,index=False,header=True)
            
           
            logging.info('data ingestion completed')
            

            #writing test and train data to next step i.e data transformation
            logging.info('writing train,test and dataframe to next step i.e data transformation')
            return(
               self.ingestion_config.test_data_file_path,
               self.ingestion_config.train_data_file_path
               
            )
            
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    ingestion_obj=DataIngestion()
    train_dataset,test_dataset=ingestion_obj.initiate_data_ingestion()
    
    #comning data transformation
    data_transformation=DataTransformation()
    train_arr,test_arr=data_transformation.initiate_data_transformation(train_dataset,test_dataset)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
