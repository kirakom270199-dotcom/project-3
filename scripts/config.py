import torch

class Config:
    # Data paths
    DISH_CSV_PATH = 'nutrition/data/dish.csv'
    INGREDIENTS_CSV_PATH = 'nutrition/data/ingredients.csv'
    IMAGES_DIR = 'nutrition/data/images'

    # Model parameters
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    # Model architecture
    TEXT_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
    IMAGE_MODEL_NAME = 'resnet50'
    TEXT_EMBEDDING_DIM = 384
    IMAGE_EMBEDDING_DIM = 2048
    HIDDEN_DIM = 512
    DROPOUT = 0.3

    # Training
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42
    EARLY_STOPPING_PATIENCE = 10

    # Logging and saving
    MODEL_SAVE_PATH = 'best_model.pth'
    LOG_DIR = 'logs/'

    def to_dict(self):
        return {key: value for key, value in self.__dict__.items()
                if not key.startswith('_')}