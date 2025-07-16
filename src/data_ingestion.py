import os
import shutil
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
from src.logger import auto_logger
from src.custom_exception import CustomException  
from config.data_ingestion_config import *

logger = auto_logger(__name__)



class DataIngestion():

    def __init__(self,data_name: str,data_dir: str, data_path: str):
        try:
            self.data_name = data_name
            self.data_dir = data_dir
            self.data_path = data_path
        except Exception as e:
            logger.error("Error at DataIngestion class initialization")    
            raise CustomException("Failed to initialize Dataingestion class", e)
        
    
    def create_dir(self):
        try:
            logger.info("Creating directory")
            raw_path = os.path.join(self.data_dir,"raw")
            if not os.path.exists(raw_path):
                os.makedirs(raw_path)
                logger.info(f"sucessfully created path: {raw_path} ")
            return raw_path
        except Exception as e:
            logger.error("Failed creating path")
            raise CustomException("path creation failed",e)


    
    def data_downloader(self,Dataname: str,destination: str):
        try:
            logger.info("Authenticating kaggle API")
            api = KaggleApi()
            api.authenticate()
            logger.info("kaggle API authenticated sucessfully")

            logger.info(f"Initializing Dataset Ingestion from Kaggle: {Dataname} ")
            api.dataset_download_files(dataset=Dataname,path=destination,unzip=True,quiet=False)
            logger.info("Downloaded dataset from kaggle sucessfully")

        except Exception as e:
            logger.error("Failed downloading dataset from kaggle")
            raise CustomException("Data Ingestion failed", e)

    def oragnize_dataset(self,destination: str):
        try:
            logger.info("Starting organizing data")
            source = "artifacts/raw/Dental OPG XRAY Dataset/Dental OPG (Classification)"
            del_dir = "artifacts/raw/Dental OPG XRAY Dataset"

            if not os.path.exists(source):
                logger.error(f"source not found at: {source}")

            target_dir = os.path.join(destination,"Dataset")

            if not os.path.exists(target_dir):
                os.makedirs(target_dir,exist_ok= True)
            
            for category in os.listdir(source):
                category_path = os.path.join(source,category)
                if os.path.isdir(category_path):
                    shutil.move(category_path,target_dir)
            logger.info(f"All files sucessfully moved to: {target_dir}")

            logger.info("Delelting unwanted data")
            shutil.rmtree(del_dir)
            logger.info("Unwanted data deleted")


        except Exception as e:
            logger.error(f"Failed organizing data set at: {destination}")
            raise CustomException("",e)



    def run_data_ingestion(self):
        try:
            path = self.create_dir()
            self.data_downloader(self.data_name,path)
            self.oragnize_dataset(path)
            logger.info("Data ingestion pipeline completed successfully")
        except Exception as e:
            logger.error(f"Error during data ingestion pipeline: {str(e)}")
            raise CustomException("Error dunring running data ingestion pipeline",e)