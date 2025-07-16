from src.logger import auto_logger
from src.data_ingestion import DataIngestion
from config.data_ingestion_config import *

logger = auto_logger(__name__)


def main():
    try:
        logger.info("Starting data ingestion process")
        data = DataIngestion(data_name, data_dir,data_path)
        data.run_data_ingestion()
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()

