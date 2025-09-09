# dataset
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class CustomDataset(Dataset):
    def __init__(self, dataframe, devices, feature_scaler=None, label_scaler=None, is_feature_extractor=False):
        self.data = dataframe
        self.is_feature_extractor = is_feature_extractor
        self.devices = devices  # Device names for load disaggregation

        # Extract Hour column as a separate feature (apply sine transform), do not standardize
        self.hour_feature = torch.tensor(np.sin(2 * np.pi * self.data[['Hour']].values / 24.0), dtype=torch.float32)

        if self.is_feature_extractor:
            # Features include all device load data
            load_features = torch.tensor(self.data[['Aggregate Load'] + self.devices].values,
                                         dtype=torch.float32)
        else:
            # 'Aggregate Load' as the input feature to be standardized
            load_features = torch.tensor(self.data[['Aggregate Load']].values, dtype=torch.float32)

        # Device loads as target outputs
        self.labels = torch.tensor(self.data[self.devices].values,
                                   dtype=torch.float32)

        # If no scaler is provided, this is training data → fit a scaler from the current data
        if feature_scaler is None:
            self.feature_scaler = StandardScaler()
            load_features = self.feature_scaler.fit_transform(load_features)
        else:
            self.feature_scaler = feature_scaler
            load_features = self.feature_scaler.transform(load_features)

        if label_scaler is None:
            self.label_scaler = StandardScaler()
            self.labels = self.label_scaler.fit_transform(self.labels)
        else:
            self.label_scaler = label_scaler
            self.labels = self.label_scaler.transform(self.labels)

        # Combine standardized load features with non-standardized Hour feature
        self.features = torch.cat((torch.tensor(load_features, dtype=torch.float32), self.hour_feature), dim=1)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    # Provide methods to get the scalers (to standardize validation/test sets)
    def get_feature_scaler(self):
        return self.feature_scaler

    def get_label_scaler(self):
        return self.label_scaler

    def get_scaled_labels(self):
        return self.labels.numpy()


class PredictDataset(Dataset):
    def __init__(self, dataframe, feature_scaler):
        self.data = dataframe

        # Use training data’s scaler to standardize "Aggregate Load" feature
        load_features = feature_scaler.transform(self.data[['Aggregate Load']].values)
        load_features = torch.tensor(load_features, dtype=torch.float32)

        # Apply sine transform to "Hour" feature, do not standardize
        hour_feature = torch.tensor(np.sin(2 * np.pi * self.data[['Hour']].values / 24.0), dtype=torch.float32)

        # Combine features
        self.features = torch.cat((load_features, hour_feature), dim=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.features[index]