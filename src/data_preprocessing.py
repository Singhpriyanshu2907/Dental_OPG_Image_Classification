from src.logger import auto_logger
from src.custom_exception import CustomException
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import cv2
import shutil
from sklearn.model_selection import train_test_split
from config.data_preprocessing_config import *


logger = auto_logger(__name__)



class DataSplitter():

    def __init__(self, category: list ,source: str,destination: str):
        try:
            logger.info("Initializing data Pre-processing class")
            self.category = category
            self.source = source
            self.destination = destination
            logger.info("Data Pre-processing class succesfully initialized")
        except Exception as e:
            logger.error("Error at data pre-processing class")
            raise CustomException("Failed to initialize data pre-processing class", e)
            


    def data_split(self):
        try:

            logger.info("making dir for train, test and val dataset")    

            # Create output directories
            os.makedirs(os.path.join(self.destination, "test"), exist_ok=True)
            os.makedirs(os.path.join(self.destination, "train"), exist_ok=True)
            os.makedirs(os.path.join(self.destination, "val"), exist_ok=True)

            logger.info("dir for train, test and val dataset created")  

            for cat in self.category:
                cat_path = os.path.join(self.source, cat)

                # Skip if directory doesn't exist
                if not os.path.exists(cat_path):
                    logger.info(f"Warning: {cat_path} not found. Skipping...")
                    continue

                images = [os.path.join(cat_path, image) for image in os.listdir(cat_path)]

                logger.info("Splitting images into train, test & val")
                # Split only if there are images
                if len(images) > 0:
                    train, temp = train_test_split(images, test_size=0.3, random_state=42)
                    val, test = train_test_split(temp, test_size=0.5, random_state=42)

                    # Copy training images
                    dest_dir_train = os.path.join(self.destination, "train", cat)
                    os.makedirs(dest_dir_train, exist_ok=True)
                    for file in train:
                        shutil.move(file, dest_dir_train)

                    # Copy validation images
                    dest_dir_val = os.path.join(self.destination, "val", cat)
                    os.makedirs(dest_dir_val, exist_ok=True)
                    for file in val:
                        shutil.move(file, dest_dir_val)

                    # Copy test images
                    dest_dir_test = os.path.join(self.destination, "test", cat)
                    os.makedirs(dest_dir_test, exist_ok=True)
                    for file in test:
                        shutil.move(file, dest_dir_test)
                else:
                    logger.info(f"Warning: No images found in {cat_path}")

            logger.info("Data Split Successfully")
        except Exception as e:
            logger.error("Failed to Split data into train & test")
            raise CustomException("Train & Test Split failed",e)
    

    def split_verifier(self):
        try:
            logger.info("Verifiying data split into train, test & val set")
            split_dir = self.destination
            clas = ["train", "val", "test"]
            classes = [classes for classes in self.category]
            for cat in clas:
                logger.info(f"\n{cat.upper()} SET:")
                for class_name in classes:
                    count = len(os.listdir(os.path.join(split_dir, cat, class_name)))
                    logger.info(f"{class_name}: {count} images")
        except Exception as e:
            logger.error("Falied to get image distribution among train, test & val set")
            raise CustomException("Falied to get image distribution among train, test & val set",e)
        
    
    def Run_DataSplitter(self):
        try:
            logger.info("Running Data_split method")     
            self.data_split()
            logger.info("Data_split method succesfully executed") 
            
            logger.info("Running split_verifier method") 
            self.split_verifier()
            logger.info("split_verifier method succesfully executed")
        
        except Exception as e:
            logger.error("Failed to excecute Run_DataSplitter Method")
            raise CustomException("Failed to excecute Run_DataSplitter Method",e)
        



class XRayPreprocessor:
    def __init__(self):
        try:
            logger.info("Initializing CLAHE for Contarcst Enhancement")    
            # OpenCV preprocessing pipeline
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            logger.info("CLAHE succesfully initialized")
        
        except Exception as e:
            logger.error("Falied to initialize CLAHE")
            raise CustomException("Falied to initialize CLAHE",e)

    
    def __call__(self, img_path):
        try:
            
            logger.info("Resizing and Scaling image")

            """Combined OpenCV + PIL preprocessing"""
            # OpenCV Processing
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, IMG_SIZE)
            img = self.clahe.apply(img)  # Contrast enhancement
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            logger.info("Image resizing and scaling completed")

            # Convert to PIL for Torchvision transforms
            return Image.fromarray(img)
        
        except Exception as e:
            logger.error("Image resizing and scaling failed")
            raise CustomException("Image resizing and scaling failed",e)
    

class DentalDataset(Dataset):
    def __init__(self, root_dir, transform=None, preprocessor=None):
        try:

            logger.info("Initailizing Pre-Processor and Tranformer")    

            self.root_dir = root_dir
            self.transform = transform
            self.preprocessor = preprocessor or XRayPreprocessor()
            self.classes = sorted(os.listdir(root_dir))
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
            self.images = self._load_images()

            logger.info("Pre-Processor and Tranformer initialized") 
        
        except Exception as e:
            logger.error("Pre-Processor and Tranformer initialization failed")
            raise CustomException("Pre-Processor and Tranformer initialization failed",e)

    def _load_images(self):
        try:
            logger.info("Loading images")
            images = []
            for cls in self.classes:
                cls_path = os.path.join(self.root_dir, cls)
                for img_name in os.listdir(cls_path):
                    img_path = os.path.join(cls_path, img_name)
                    images.append((img_path, self.class_to_idx[cls]))
            logger.info("Images loaded succesfully")
            return images

        except Exception as e:
            logger.error("Image loading failed")
            raise CustomException("Image loading failed",e)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
                
            img_path, label = self.images[idx]

            logger.info("Applying pre-processing on image")
            image = self.preprocessor(img_path)
            logger.info("Pre-processing applied succesfully")

            logger.info("Applying Image transformation and augmentation")
            if self.transform:
                image = self.transform(image)
            logger.info("Image Augmentation and tranformation applied succesfully")

            return image, label
        
        except Exception as e:
            logger.error("")
            raise CustomException("",e) 


def get_transforms():
    try:

        logger.info("Transforming Train Set")

        """Different transforms for train/val/test"""
        train_transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        logger.info("Train Set Transformed Successfully")

        logger.info("Transforming Test & Val Set")

        test_transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        logger.info("Test & Val Set Transformed Successfully")

        return train_transform, test_transform
    
    except Exception as e:
        logger.error("Transforming Test, Train & Val set failed")
        raise CustomException("Transforming Test, Train & Val set failed",e)