import logging 
import os
from datetime import datetime

# how the log file should get created i.e name of log file
LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
#path to where the logs files created i.e current working directory
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)
#eventhough if there is file or folder we keep appending the file or folder
os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    #format how the message will get printed
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s -%(message)s",
    level=logging.INFO,
)

if __name__=="__main__":
    logging.info("Logging has started")