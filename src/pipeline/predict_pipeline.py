import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_obj

class Predictpipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\preprocessor.pkl'
            #load_object function in utils will import file and load it
            model=load_obj(file_path=model_path)
            preprocessor=load_obj(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
        
#custom data class is responsible in mapping the values given in html page to backed values

class CustomData:
    def __init__(self,
                Age:int,
                Sex: str,
                chest_pain_type: str,
                BP:int,
                Cholesterol:str,
                FBS_over_120:str,
                EKG_results:str,
                Max_HR:int,
                Exercise_angina:str,
                ST_depression:int,
                Slope_of_ST:str,
                Number_of_vessels_fluro:int,
                Thallium:str):
        
        self.Age=Age
        self.Sex=Sex
        self.chest_pain_type=chest_pain_type
        self.BP=BP
        self.Cholesterol=Cholesterol
        self.FBS_over_120=FBS_over_120
        self.EKG_results=EKG_results
        self.Max_HR=Max_HR
        self.Exercise_angina=Exercise_angina
        self.ST_depression=ST_depression
        self.Slope_of_ST=Slope_of_ST
        self.Number_of_vessels_fluro=Number_of_vessels_fluro
        self.Thallium=Thallium
 # this calss will return all the input in the form of dataframe beacause we train the model in the form of dataframe

    def get_data_as_frame(self):
        try:
            custom_data_input_dict={
                "Age":[self.Age],
                "Sex":[self.Sex],
                "chest_pain_type":[self.chest_pain_type],
                "BP":[self.BP],
                "Cholesterol":[self.Cholesterol],
                "FBS_over_120":[self.FBS_over_120],
                "EKG_results":[self.EKG_results],
                "Max_HR":[self.Max_HR],
                "Exercise_angina":[self.Exercise_angina],
                "ST_depression":[self.ST_depression],
                "Slope_of_ST":[self.Slope_of_ST],
                "Number_of_vessels_fluro":[self.Number_of_vessels_fluro],
                "Thallium":[self.Thallium],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)