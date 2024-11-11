import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self, channels, time_second, freq):
        super(FeatureExtractor, self).__init__()
        self.channels = channels
        self.time_second = time_second
        self.freq = freq

        self.small_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=49, stride=6, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=16, stride=16),
            nn.Dropout(0.5),
            nn.Conv1d(32, 64, kernel_size=7, stride=1, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=7, stride=1, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=7, stride=1, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8)
        )

        self.large_cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=399, stride=50, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(0.5),
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )

    def forward(self, x):
        #print("input",x.shape)#40 (60) 19 (100)6000
        batch_size, time_step, channels, freq = x.size()
        x = x.view(batch_size * channels, 1, freq * time_step)# Reshape to (batch_size * channels, 1, freq*time_step)
        features_small = self.small_cnn(x)
        features_large = self.large_cnn(x)
        features_small = features_small.view(batch_size, channels, -1)
        features_large = features_large.view(batch_size, channels, -1)
        features = torch.cat((features_small, features_large), dim=2)
        return features

class FeatureNet(nn.Module):
    def __init__(self, channels=19, time_second=60, freq=100, num_class = 1):
        super(FeatureNet, self).__init__()
        self.extractor = FeatureExtractor(channels, time_second, freq)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(12160, 128)  # Adjust the input features according to the output of FeatureExtractor uesfft=0:24320, uesfft=1: 12160
        self.fc2 = nn.Linear(128, num_class)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        #print("before",x.shape)
        x = self.extractor(x)
        #print("afterextract",x.shape)
        x = self.flatten(x)
        #print("afterflatten",x.shape)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        #print("fc1",x.shape)
        x = self.fc2(x)
        #print("output",x.shape)
        return x

