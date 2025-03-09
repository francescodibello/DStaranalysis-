import torch
from torch.utils.data import Dataset
import uproot
import numpy as np

class ParticleJetDataset(Dataset):
    def __init__(self, root_file):
        self.tree = uproot.open(root_file)["tree"]  # Change "tree" to your actual TTree name

        # Load particle-level variables
        self.part_eta = self.tree["part_eta"].arrays(library="np")
        self.part_phi = self.tree["part_phi"].arrays(library="np")
        self.part_mass = self.tree["part_mass"].arrays(library="np")
        self.part_massReco = self.tree["part_massReco"].arrays(library="np")
        self.part_pid = self.tree["part_pid"].arrays(library="np")
        self.part_isFromD = self.tree["part_isFromD"].arrays(library="np")
        self.part_isFromDStar = self.tree["part_isFromDStar"].arrays(library="np")

        # Load jet-level variable (c-jet label: at least one D* in the jet)
        self.jet_isCJet = np.array([
            1 if np.any(self.part_isFromDStar[idx] > 0) else 0
            for idx in range(len(self.part_eta))
        ], dtype=np.float32)

        self.num_jets = len(self.part_eta)

    def __len__(self):
        return self.num_jets

    def __getitem__(self, idx):
        # Get per-particle features
        part_features = np.stack([
            self.part_eta[idx],
            self.part_phi[idx],
            self.part_mass[idx],
            self.part_massReco[idx],
            self.part_pid[idx]
        ], axis=1)

        # Get per-particle labels
        labels_particle = np.stack([
            self.part_isFromD[idx],
            self.part_isFromDStar[idx]
        ], axis=1)

        # Get per-jet label (1 if c-jet, 0 otherwise)
        label_jet = self.jet_isCJet[idx]

        return (torch.tensor(part_features, dtype=torch.float32), 
                torch.tensor(labels_particle, dtype=torch.float32),
                torch.tensor(label_jet, dtype=torch.float32))

