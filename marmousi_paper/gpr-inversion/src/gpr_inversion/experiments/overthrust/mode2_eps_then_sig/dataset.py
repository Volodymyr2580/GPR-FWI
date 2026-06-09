"""Dataset helpers for the OverThrust mode2 epsilon-then-sigma experiment."""

import torch.utils.data as data_utils


class ShotDataset(data_utils.Dataset):
    """Small wrapper that keeps observed data, source geometry, and receivers together."""

    def __init__(self, seismic_data, source_locations, receiver_locations):
        self.seismic_data = seismic_data
        self.source_locations = source_locations
        self.receiver_locations = receiver_locations

    def __getitem__(self, item):
        return (
            self.seismic_data[item],
            self.source_locations[item],
            self.receiver_locations[item],
        )

    def __len__(self):
        return len(self.seismic_data)
