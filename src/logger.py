import os
import logging
from datetime import datetime


log_dir = "logs"

if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok= True)

log_file = os.path.join(log_dir,f"log-{datetime.now().strftime('%y-%m-%d')}.log")

logging.basicConfig(
    filename = log_file,
    level = logging.INFO,
    format = "[%(asctime)s] %(levelname)s - %(message)s"
)


def auto_logger(name):
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    return logger




