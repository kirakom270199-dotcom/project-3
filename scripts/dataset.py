import pandas as pd
import numpy as np
from PIL import Image
import os
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

class FoodDataset(Dataset):
    def __init__(self, df, ingredients_df, images_dir, transform=None, is_train=True, config=None):
        self.df = df
        self.ingredients_df = ingredients_df
        self.images_dir = images_dir
        self.transform = transform
        self.is_train = is_train
        self.config = config

        # Initialize tokenizer for text
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

        # Prepare ingredients mapping
        self.ingredients_dict = dict(zip(ingredients_df['id'], ingredients_df['ingr']))

        # Filter out samples without images
        self.valid_indices = []
        for idx in range(len(df)):
            dish_id = df.iloc[idx]['dish_id']
            img_path = os.path.join(images_dir, dish_id, 'rgb.png')
            if os.path.exists(img_path):
                self.valid_indices.append(idx)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        row = self.df.iloc[actual_idx]

        dish_id = row['dish_id']
        total_calories = row['total_calories']
        total_mass = row['total_mass']
        ingredients_ids = row['ingredients']

        # Load and process image
        img_path = os.path.join(self.images_dir, dish_id, 'rgb.png')
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)

        if self.transform:
            image = self.transform(image=image)['image']

        # Process ingredients text
        ingredients_list = ingredients_ids.split(';')
        ingredients_text = []
        for ingr_id in ingredients_list:
            if ingr_id in self.ingredients_dict:
                ingredients_text.append(self.ingredients_dict[ingr_id])

        ingredients_text = ' '.join(ingredients_text)

        # Tokenize ingredients text
        text_inputs = self.tokenizer(
            ingredients_text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        return {
            'image': image,
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'total_mass': torch.tensor(total_mass, dtype=torch.float32),
            'calories': torch.tensor(total_calories, dtype=torch.float32),
            'dish_id': dish_id
        }

def get_transforms(config, is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(*config.IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(*config.IMAGE_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

def create_data_loaders(df_dish, df_ingredients, config):
    # Split data
    train_df = df_dish[df_dish['split'] == 'train'].reset_index(drop=True)
    test_df = df_dish[df_dish['split'] == 'test'].reset_index(drop=True)

    # Create datasets
    train_dataset = FoodDataset(
        train_df, df_ingredients, config.IMAGES_DIR,
        transform=get_transforms(config, is_train=True),
        is_train=True, config=config
    )

    test_dataset = FoodDataset(
        test_df, df_ingredients, config.IMAGES_DIR,
        transform=get_transforms(config, is_train=False),
        is_train=False, config=config
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=4, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, test_loader, train_dataset, test_dataset