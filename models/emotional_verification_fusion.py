import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.model_editing import drop_layers, last_linear_layer_dimensions, model_no_grad


class CNNFusionModel(nn.Module):
    def __init__(self, height, width, proj=256, first_channels=32, kernel_size=5,
                 emotional_model=None, layers_dropped=1, emotional_training=False):
        super(CNNFusionModel, self).__init__()
        self.num_cnn_layers = 3
        self.cnn_channels = 2
        self.height = height
        self.width = width
        self.first_channels = first_channels

        # Emotional layer details
        if emotional_training is False:
            self.emotional_model = model_no_grad(emotional_model)
        else:
            self.emotional_model = emotional_model

        # Cut last emotional features
        self.emotional_model = drop_layers(model=emotional_model,
                                           layers_dropped=layers_dropped)
        # Get dimensions
        _, emotional_out_dim = last_linear_layer_dimensions(
            self.emotional_model)

        # Speaker Verification Layers
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,
                      first_channels,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=2),
            nn.BatchNorm2d(first_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(first_channels,
                      self.cnn_channels*first_channels,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=2),
            nn.BatchNorm2d(self.cnn_channels*first_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(2*first_channels,
                      (self.cnn_channels**2) * first_channels,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=2),
            nn.BatchNorm2d((self.cnn_channels**2)*first_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.linear_layer1 = nn.Sequential(
            nn.Dropout2d(0.75),
            nn.Linear(self.calc_out_size(), 1024),
            nn.LeakyReLU()
        )

        self.linear_layer2 = nn.Sequential(
            nn.Dropout(0.5),
            # include emotional content
            nn.Linear(1024 + emotional_out_dim, proj),
            nn.LeakyReLU()
        )

    def forward(self, x):
        # x.size() = (N*M, 1, max_seq, fe)
        # CNN
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)

        # DNN -- pass through linear layers
        out = self.linear_layer1(out)

        # Emotion labels
        emotional_embeddings = self.emotional_model(x)

        # Concatenate emotional and speaker discriminative information
        out = torch.cat((out, emotional_embeddings), dim=1)
        # Last forward pass using emotional + SV information
        # (1024 + #emot_embedding=256) -> 256
        out = self.linear_layer2(out)

        return out

    def calc_out_size(self):
        height = int(self.height / 8)
        width = int(self.width / 8)
        kernels = self.cnn_channels * \
            (self.num_cnn_layers - 1) * self.first_channels
        return kernels * height * width

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
