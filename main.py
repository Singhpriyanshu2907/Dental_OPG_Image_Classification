from src.logger import auto_logger
from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataSplitter
from src.model_trainer import *
from config.data_ingestion_config import *
from config.data_preprocessing_config import *
from config.model_trainer_config import *

logger = auto_logger(__name__)


def main():
    try:
        # logger.info("Starting data ingestion process")
        # data = DataIngestion(data_name, data_dir,data_path)
        # # data.run_data_ingestion()
        # logger.info("Data ingestion process succesfully excecuted")



        # logger.info("Starting data Splitting process")
        # data = DataSplitter(category, source, destination)
        # data.Run_DataSplitter()
        # logger.info("Data Splitting process succesfully excecuted, Splitted data into Train, Test and Val Set")

        models_to_compare = [
        "resnet18"]


        # Create the trainer instance
        trainer = ComparativeTrainer(models_to_compare)

        # Call the new method to train all models with the fine-tuning strategy
        trainer.train_all_fine_tuned()

        # Save the new results
        import json
        with open("model_comparison_finetuned.json", "w") as f:
            json.dump(trainer.results, f, indent=2)



    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()

