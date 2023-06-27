import sys
from src.logger import logging

'''whenever error raises we call "error_message_details" function '''
# whenver there is exception gets raised we push our own custom message
def error_message_details(error,error_detail:sys):
    #_,_,exc_tb will capture the details in which file exceptin occured and which line exception occured
    _,_,exc_tb=error_detail.exc_info()
    #to get the file name where exception is occuring
    file_name=exc_tb.tb_frame.f_code.co_filename
    #format to print exception error in a file
    error_message="Exception occured in python scroipt [{0}] line number[{1}] error message[{2}]".format(
    file_name,exc_tb.tb_lineno,str(error))#exc_tb.tb_lineno = line number
    return error_message

class CustomException(Exception):
    #constructor
    def __init__(self,error_message,error_detail:sys):
        #since we are inheriting from exception to inherit the init function
        super().__init__(error_message)
        #calling the function
        self.error_message=error_message_details(error_message,error_detail=error_detail)
    #inherting __str__ functionlity from custom exception
    #when we raise CustomException it print error message
    def __str__(self):
        return self.error_message