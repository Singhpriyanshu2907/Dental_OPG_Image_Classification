from src.logger import auto_logger
from src.custom_exception import CustomException
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter
from src.data_preprocessing import *
from config.model_trainer_config import *
import torch.optim as optim
import torch.autograd.grad_mode as grad_mode
import timm
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader



logger = auto_logger(__name__)



class ModelZoo:
    @staticmethod
    def get_model(model_name, NUM_CLASSES):
        try:
            logger.info("Initializing models")    
            """Factory method for pretrained models"""
            if model_name == "resnet18":
                model = models.resnet18(weights='DEFAULT')
                model.fc = nn.Linear(512, NUM_CLASSES)
            # elif model_name == "efficientnet_b0":
            #     model = models.efficientnet_b0(weights='DEFAULT')
            #     model.classifier[1] = nn.Linear(1280, NUM_CLASSES)
            # elif model_name == "densenet121":
            #     model = models.densenet121(weights='DEFAULT')
            #     model.classifier = nn.Linear(1024, NUM_CLASSES)
            # elif model_name == "convnext_tiny":
            #     model = models.convnext_tiny(weights='DEFAULT')
            #     model.classifier[2] = nn.Linear(768, NUM_CLASSES)
            # elif model_name == "swin_t":
            #     model = models.swin_t(weights='DEFAULT')
            #     model.head = nn.Linear(768, NUM_CLASSES)
            # elif model_name == "vit_b16":
            #     model = timm.create_model("vit_base_patch16_224", pretrained=True)
            #     model.head = nn.Linear(768, NUM_CLASSES)
            else:
                raise ValueError(f"Unknown model: {model_name}")
            logger.info("Models initialized")

            return model
        except Exception as e:
            logger.error("Failed to initialize models")
            CustomException("Failed to initialize models",e)

    @staticmethod
    def set_parameter_groups(model, backbone_lr, classifier_lr):
        """
        Sets up parameter groups for differential learning rates.
        Returns a list of dictionaries for the optimizer.
        """
        try:
            logger.info("Initializing parameter setter")
            # Find the name of the final classifier layer
            if hasattr(model, 'fc'):
                classifier_name = 'fc'
            elif hasattr(model, 'classifier'):
                classifier_name = 'classifier'
            elif hasattr(model, 'head'):
                classifier_name = 'head'
            else:
                raise ValueError("Could not find a classifier/head layer.")

            # Separate parameters for the classifier and the rest of the model (backbone)
            classifier_params = []
            backbone_params = []

            for name, param in model.named_parameters():
                if classifier_name in name:
                    classifier_params.append(param)
                else:
                    backbone_params.append(param)
            
            logger.info("Parameter setter Initialized")

            return[{'params': backbone_params, 'lr': backbone_lr},{'params': classifier_params, 'lr': classifier_lr}]
        
        except Exception as e:
            logger.error("Failed to set parameters")
            raise CustomException("Failed to set parameters",e) 

class ComparativeTrainer:
    def __init__(self, model_names):
        try:
            logger.info("Initializing ComparativeTrainer...")

                        # Create directories to save models
            os.makedirs(WARMUP_MODEL_PATH, exist_ok=True)
            os.makedirs(FINETUNE_MODEL_PATH, exist_ok=True)
            os.makedirs(OVERALL_BEST_MODEL_PATH, exist_ok=True)
            logger.info("Model directories created.")

            self.models = {
                name: ModelZoo.get_model(name, NUM_CLASSES).to(DEVICE)
                for name in model_names
            }
            self.results = {}
            self._init_dataloaders()

            # New attributes to track the overall best model
            self.overall_best_accuracy = 0.0
            self.overall_best_model_name = None
            logger.info("ComparativeTrainer initialized successfully.")
        
        except Exception as e:
            logger.error(f"Error during trainer initialization: {e}")
            raise CustomException("",e)
        

    def _init_dataloaders(self):
        try:

            logger.info("Initializing data loaders...")

            train_transform, test_transform = get_transforms()
            preprocessor = XRayPreprocessor()

            self.train_ds = DentalDataset(TRAIN_PATH, train_transform, preprocessor)
            self.val_ds = DentalDataset(VAL_PATH, test_transform, preprocessor)
            self.test_ds = DentalDataset(TEST_PATH, test_transform, preprocessor)

            self.train_loader = DataLoader(self.train_ds, batch_size=BATCH_SIZE, shuffle=True)
            self.val_loader = DataLoader(self.val_ds, batch_size=BATCH_SIZE)
            self.test_loader = DataLoader(self.test_ds, batch_size=BATCH_SIZE)

            logger.info("Data loaders initialized.")
        
        except Exception as e:
            logger.error(f"Error during Data Loader initialization: {e}")
            raise CustomException("Error during Data Loader initialization",e)



    def _set_requires_grad(self, model, requires_grad):
        """Sets the requires_grad attribute for all parameters in a model."""
        for param in model.parameters():
            param.requires_grad = requires_grad



    def _freeze_all_layers(self, model, unfreeze_classifier=False):
        """Freezes all layers and optionally unfreezes the classifier."""
        for param in model.parameters():
            param.requires_grad = False

        if unfreeze_classifier:
            if hasattr(model, 'fc'):
                for param in model.fc.parameters(): param.requires_grad = True
            elif hasattr(model, 'classifier'):
                for param in model.classifier.parameters(): param.requires_grad = True
            elif hasattr(model, 'head'):
                for param in model.head.parameters(): param.requires_grad = True




    def _train_step(self, model, optimizer, criterion, scheduler, loader):
        """A single training epoch."""
        model.train()
        train_loss = 0.0
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()
        return train_loss / len(loader)
    



    def train_model(self, model, name, num_epochs, lr):
        """Trains the model with a fixed learning rate for all trainable parameters."""

        try:

            logger.info(f"Freezing all layers except classifier for warm-up of model: {name}")

            self._freeze_all_layers(model, unfreeze_classifier=True)

            # Only pass trainable parameters to the optimizer
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
            criterion = nn.CrossEntropyLoss()

            best_val_acc = 0.0
            history = {"train_loss": [], "val_acc": []}

            log_path = os.path.join(TENSORBOARD_LOG_PATH, f"{name}_warmup")
            writer = SummaryWriter(log_path)


            print(f"Starting Warm-up Training for {name} for {num_epochs} epochs...")
            logger.info(f"Starting Warm-up Training for {name} for {num_epochs} epochs...")

            for epoch in range(num_epochs):
                train_loss = self._train_step(model, optimizer, criterion, scheduler, self.train_loader)
                val_acc = self.evaluate(model, self.val_loader)
                history["train_loss"].append(train_loss)
                history["val_acc"].append(val_acc)

                # Log metrics to TensorBoard
                writer.add_scalar('Loss/train_warmup', train_loss, epoch)
                writer.add_scalar('Accuracy/val_warmup', val_acc, epoch)

                print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
                logger.info(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")


                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), os.path.join(WARMUP_MODEL_PATH, f"best_{name}_warmup.pth"))
                    logger.info(f"New best validation accuracy found. Saved best_warmup model.")
            
            writer.close()

            return history
        
        except Exception as e:
            logger.error(f"Failed to log warmup model {e}")
            raise CustomException("Failed to log warmup model",e) 
        



    def train_and_fine_tune(self, model_name, warmup_epochs=1, fine_tune_epochs=1,
                            warmup_lr=1e-3, fine_tune_backbone_lr=1e-5, fine_tune_classifier_lr=1e-4):
        
        try:

            """Two-stage training: warm-up then fine-tuning."""
            model = self.models[model_name]


            print(f"\n{'='*50}\nTraining and Fine-Tuning {model_name}\n{'='*50}")
            logger.info(f"\n{'='*50}\nStarting training for {model_name}\n{'='*50}")

            # Stage 1: Warm-up (Classifier-only training)
            warmup_history = self.train_model(model, model_name, warmup_epochs, warmup_lr)
            

            warmup_path = os.path.join(WARMUP_MODEL_PATH, f"best_{model_name}_warmup.pth")
            
            if not os.path.exists(warmup_path):
                torch.save(model.state_dict(), warmup_path)
            
            logger.info(f"Loading best warm-up model state from {warmup_path}")
            model.load_state_dict(torch.load(warmup_path))


            # Stage 2: Fine-Tuning (Entire model training)
            print(f"\nStarting Fine-Tuning for {model_name} for {fine_tune_epochs} epochs...")
            logger.info(f"Starting Fine-Tuning for {model_name} for {fine_tune_epochs} epochs...")

            logger.info("Unfreezing all layers for fine-tuning.")
            self._set_requires_grad(model, requires_grad=True)

            param_groups = ModelZoo.set_parameter_groups(
                model, fine_tune_backbone_lr, fine_tune_classifier_lr
            )

            logger.info(f"Using differential learning rates: backbone_lr={fine_tune_backbone_lr}, classifier_lr={fine_tune_classifier_lr}")
            optimizer = optim.AdamW(param_groups)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=fine_tune_epochs)
            criterion = nn.CrossEntropyLoss()

            best_val_acc = self.evaluate(model, self.val_loader)
            history = {"train_loss": warmup_history["train_loss"], "val_acc": warmup_history["val_acc"]}

            
            finetune_path = os.path.join(FINETUNE_MODEL_PATH, f"best_{model_name}_finetune.pth")

            if not os.path.exists(finetune_path):
                torch.save(model.state_dict(), finetune_path)
                logger.info(f"Initial fine-tune model state saved to {finetune_path}")

                log_path = os.path.join(TENSORBOARD_LOG_PATH, f"{model_name}_finetune")
                writer = SummaryWriter(log_path)


            for epoch in range(fine_tune_epochs):
                train_loss = self._train_step(model, optimizer, criterion, scheduler, self.train_loader)
                val_acc = self.evaluate(model, self.val_loader)
                history["train_loss"].append(train_loss)
                history["val_acc"].append(val_acc)

                writer.add_scalar('Loss/train_finetune', train_loss, epoch)
                writer.add_scalar('Accuracy/val_finetune', val_acc, epoch)

                print(f"Fine-tune Epoch  {epoch+1}/{fine_tune_epochs} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
                logger.info(f"Fine-tune Epoch {epoch+1}/{fine_tune_epochs} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), finetune_path)
                    logger.info(f"New best validation accuracy found. Saved best_finetune model.")


            final_model_path = os.path.join(FINETUNE_MODEL_PATH, f"final_{model_name}_finetune.pth")
            torch.save(model.state_dict(), final_model_path)
            logger.info(f"Final model state saved to {final_model_path}")

            test_metrics = self.test_model(model, name=f"{model_name}_finetune")
            self.results[model_name] = {"history": history, "test_metrics": test_metrics}
            logger.info(f"Testing complete for {model_name}. Accuracy: {test_metrics['accuracy']:.4f}")


            writer.close()
        
        except Exception as e:
            logger.error(f"Failed to train and finetune model {e}")
            raise CustomException("Failed to train and finetune model",e)

    
    
    
    def evaluate(self, model, loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    
    
    
    def test_model(self, model, name):

        try:

            best_model_path = os.path.join(FINETUNE_MODEL_PATH, f"best_{name}.pth")
            final_model_path = os.path.join(FINETUNE_MODEL_PATH, f"final_{name}.pth")
            
            if not os.path.exists(best_model_path):
                logger.warning(f"Best model file '{best_model_path}' not found. Loading final model instead.")
                model_path = final_model_path
            else:
                model_path = best_model_path

            logger.info(f"Loading model state from {model_path} for testing.")
            model.load_state_dict(torch.load(model_path))
            model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for inputs, labels in self.test_loader:
                    inputs = inputs.to(DEVICE)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            return classification_report(all_labels, all_preds,
                                        target_names=self.test_ds.classes,
                                        output_dict=True)
        except Exception as e:
            logger.error(f"Failed to test model {e}")
            raise CustomException("Failed to test model",e)

    
    
    def train_all_fine_tuned(self):

        try:

            logger.info("Starting training for all models...")
            # New variable to track the overall best model
            best_overall_accuracy = 0.0
            best_overall_model_name = ""

            for name in self.models.keys():
                self.train_and_fine_tune(name)

                # Check if this model is the best so far
                current_accuracy = self.results[name]['test_metrics']['accuracy']
                if current_accuracy > best_overall_accuracy:
                    best_overall_accuracy = current_accuracy
                    best_overall_model_name = name
                    logger.info(f"New overall best model found: {name} with accuracy: {current_accuracy:.4f}")

            # Print comparative results
            print("\n\n=== Final Test Accuracy (after fine-tuning) ===")
            logger.info("\n\n=== Final Test Accuracy (after fine-tuning) ===")


            for name, result in self.results.items():
                print(f"{name}: {result['test_metrics']['accuracy']:.4f}")
                logger.info(f"{name}: {result['test_metrics']['accuracy']:.4f}")

            # Save the single best-performing model
            if best_overall_model_name:
                print(f"\nSaving best overall model: {best_overall_model_name} with accuracy: {best_overall_accuracy:.4f}")
                logger.info(f"\nSaving best overall model: {best_overall_model_name} with accuracy: {best_overall_accuracy:.4f}")
                

                best_model_path = os.path.join(FINETUNE_MODEL_PATH, f"best_{best_overall_model_name}_finetune.pth")
                final_save_path = os.path.join(OVERALL_BEST_MODEL_PATH, "best_overall_model.pth")

                # Load the best model's state and save it
                best_model_state = torch.load(best_model_path)
                torch.save(best_model_state, final_save_path)
                
                print(f"Model saved to {final_save_path}")
                logger.info(f"Overall best model saved to {final_save_path}")
        
        except Exception as e:
            logger.error(f"No best overall model found to save {e}")
            raise CustomException("No best overall model found to save",e) 