import torch
import torch.nn as nn

class ParticleJetClassifier(nn.Module):
    def __init__(self, input_dim=3, embed_dim=128, num_heads=4, num_layers=5, hidden_dim=128):
        super(ParticleJetClassifier, self).__init__()

        self.embedding = nn.Linear(input_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.particle_fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 3),
        )

        self.jet_fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)

        particle_out = self.particle_fc(x)
        jet_representation = torch.mean(x, dim=1)
        jet_out = self.jet_fc(jet_representation)

        return particle_out, jet_out

