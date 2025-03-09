import torch
from torch.utils.data import Dataset
import uproot
import numpy as np

class ParticleJetDataset(Dataset):
    def __init__(self, root_file, reduce_ds=0):
        self.tree = uproot.open(root_file)["tree"]


        reduce_ds=2
        # Get number of events
        self.nevents = self.tree.num_entries
        if  reduce_ds == 0:
            self.nevents = self.nevents
        elif reduce_ds >= 1.0:
            self.nevents = int(reduce_ds)

        print(f"Loading dataset with {self.nevents} events...")

        # Define Variables
        self.particle_variables = ['part_eta', 'part_phi', 'part_mass', 'part_massReco', 'part_pid']
        self.particle_labels = ['part_isFromD', 'part_isFromDStar']
        self.jet_variables = ['jet_energy', 'jet_eta']

        # Load particle-level data
        self.full_data_array = {}
        for var in self.particle_variables + self.particle_labels:
            self.full_data_array[var] = self.tree[var].array(library="np", entry_stop=self.nevents)

        # Load jet-level data
        for var in self.jet_variables:
            self.full_data_array[var] = self.tree[var].array(library="np", entry_stop=self.nevents)

        # Convert uproot output to correct format
        for key in self.full_data_array.keys():
            if isinstance(self.full_data_array[key], dict):
                self.full_data_array[key] = list(self.full_data_array[key].values())
            else:
                self.full_data_array[key] = np.array(self.full_data_array[key])

        # Define jet-level label (c-jet: at least one D* particle)
        self.jet_isCJet = np.array([
            1 if bool(np.any(self.full_data_array["part_isFromDStar"][idx] > 0)) else 0
            for idx in range(self.nevents)
        ], dtype=np.float32)

    def __len__(self):
        return self.nevents

    def __getitem__(self, idx):
        # Extract per-particle features
        part_features = np.stack([
            self.full_data_array["part_eta"][idx],
            self.full_data_array["part_phi"][idx],
            self.full_data_array["part_mass"][idx],
            self.full_data_array["part_massReco"][idx],
            self.full_data_array["part_pid"][idx]
        ], axis=1)

        # Extract per-particle labels
        labels_particle = np.stack([
            self.full_data_array["part_isFromD"][idx],
            self.full_data_array["part_isFromDStar"][idx]
        ], axis=1)

        # Extract jet-level label
        label_jet = self.jet_isCJet[idx]

        return (torch.tensor(part_features, dtype=torch.float32),
                torch.tensor(labels_particle, dtype=torch.float32),
                torch.tensor(label_jet, dtype=torch.float32))

