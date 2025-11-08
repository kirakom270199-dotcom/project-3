import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import timm

class FoodCaloriePredictor(nn.Module):
    def __init__(self, config):
        super(FoodCaloriePredictor, self).__init__()
        self.config = config

        # Image encoder (ResNet50)
        self.image_encoder = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0  # Remove classification head
        )

        # Text encoder (Sentence Transformer)
        self.text_encoder = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)

        # Freeze encoders initially
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Fusion layers
        self.fusion_input_dim = config.IMAGE_EMBEDDING_DIM + config.TEXT_EMBEDDING_DIM + 1  # +1 for total_mass

        self.fusion_layers = nn.Sequential(
            nn.Linear(self.fusion_input_dim, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM // 2, 1)
        )

    def forward(self, image, input_ids, attention_mask, total_mass):
        # Image features
        image_features = self.image_encoder(image)

        # Text features (using [CLS] token)
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token

        # Concatenate features
        combined_features = torch.cat([
            image_features,
            text_features,
            total_mass.unsqueeze(1)
        ], dim=1)

        # Predict calories
        calories_pred = self.fusion_layers(combined_features).squeeze(1)

        return calories_pred

    def unfreeze_encoders(self):
        """Unfreeze encoder layers for fine-tuning"""
        for param in self.image_encoder.parameters():
            param.requires_grad = True
        for param in self.text_encoder.parameters():
            param.requires_grad = True