"""
Architecture:
  cnn: Conv2d(N_CH, 32, 3, padding=1) -> ReLU -> MaxPool2d(2)
       -> Conv2d(32, 64, 3, padding=1) -> ReLU
       -> Conv2d(64, 64, 3, padding=1) -> ReLU -> Flatten
  fc:  Linear(cnn_flat + n_global, features_dim) -> ReLU

With patch_radius=12 (patch_size=25): 64*12*12 + 5 = 9221 -> 256.
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CostmapCNNExtractor(BaseFeaturesExtractor):
    """CNN + global-scalar feature extractor for costmap routing."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim=features_dim)

        patch_space = observation_space["patch"]    # (N_CHANNELS, H, W)
        global_space = observation_space["global"]  # (N_GLOBAL,)

        n_in = patch_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_in, 32, kernel_size=3, padding=1),  # idx 0
            nn.ReLU(),                                      # idx 1
            nn.MaxPool2d(2),                                  # idx 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),     # idx 3
            nn.ReLU(),                                        # idx 4
            nn.Conv2d(64, 64, kernel_size=3, padding=1),     # idx 5
            nn.ReLU(),                                        # idx 6
            nn.Flatten(),                                     # idx 7
        )

        with torch.no_grad():
            cnn_flat_dim = self.cnn(torch.zeros(1, *patch_space.shape)).shape[1]

        fc_in_dim = cnn_flat_dim + global_space.shape[0]
        self.fc = nn.Sequential(
            nn.Linear(fc_in_dim, features_dim),  # idx 0
            nn.ReLU(),                            # idx 1
        )

    def forward(self, observations: dict) -> torch.Tensor:
        cnn_out = self.cnn(observations["patch"])
        combined = torch.cat([cnn_out, observations["global"]], dim=1)
        return self.fc(combined)
