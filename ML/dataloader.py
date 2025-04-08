import torch
from torch.utils.data import Dataset
import uproot
import numpy as np

from vertexing import fit

class ParticleJetDataset(Dataset):
    def __init__(self, root_file, reduce_ds=0):
        self.tree = uproot.open(root_file)["tree"]


        reduce_ds=1
        # Get number of events
        self.nevents = self.tree.num_entries
        if  reduce_ds == 0:
            self.nevents = self.nevents
        elif reduce_ds >= 1.0:
            self.nevents = int(reduce_ds)

        print(f"Loading dataset with {self.nevents} events...")

        # Define Variables
        self.particle_variables = ['part_eta', 'part_phi', 'part_d0val', 'part_dzval', 'part_mass', 'part_massReco', 'part_pid']
        self.particle_labels = ['part_isFromD', 'part_isFromDStar']
        self.jet_variables = ['jet_energy', 'jet_eta']

        # Load particle-level data
        self.full_data_array = {}
        for var in self.particle_variables + self.particle_labels:
            self.full_data_array[var] = self.tree[var].array(library="np", entry_stop=self.nevents)

        # Building tracks' origins
        origins = []
        for phi, d0, z0 in zip(self.full_data_array["part_phi"], self.full_data_array["part_d0val"], self.full_data_array["part_dzval"]):
            y0 = abs(d0)*np.sin(phi)
            x0 = abs(d0)*np.cos(phi)
            origins.append([x0, y0, z0])
        self.full_data_array["part_origin"] = np.array(origins)

        # Building tracks' versors
        versors = []
        for eta, phi in zip(self.full_data_array["part_eta"], self.full_data_array["part_phi"]):
            theta = 2*np.arctan(np.exp(-eta))
            ax = np.sin(theta)*np.cos(phi)
            ay = np.sin(theta)*np.sin(phi)
            az = np.cos(theta)
            versors.append([ax,ay,az])
        self.full_data_array["part_versor"] = np.array(versors)

        # Best chi2
        best_chi2s = []
        fit_iter = 100
        for o, a, l in zip(self.full_data_array["part_origin"], self.full_data_array["part_versor"], self.full_data_array["part_isFromD"]):
            fit_o = torch.tensor(o[l==1])
            fit_a = torch.tensor(a[l==1])
            
            chi2, _ = fit(fit_o, fit_a, None, fit_iter)
            best_chi2s.append(chi2.item())
        self.full_data_array["best_chi2"] = np.array(best_chi2s)
            


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
            #self.full_data_array["part_mass"][idx],
            #self.full_data_array["part_massReco"][idx],
            #self.full_data_array["part_pid"][idx],
            #target
            self.full_data_array["part_isFromD"][idx]
        ], axis=1)


        # **Convert one-hot labels to three-class indices**
        isFromD = self.full_data_array["part_isFromD"][idx]
        isFromDStar = self.full_data_array["part_isFromDStar"][idx]

        # Convert to class index:
        # Class 2: D* meson
        # Class 1: D meson
        # Class 0: Everything else (background)
        labels_particle = np.full_like(isFromD, 0)  # Default to "Anything Else" (Class 0)
        labels_particle[isFromD > 0] = 1  # Class 1: D meson
        labels_particle[isFromDStar > 0] = 2  # Class 2: D* meson (overwrites D if both exist)

        # Extract jet-level label
        label_jet = self.jet_isCJet[idx]

        return (torch.tensor(part_features, dtype=torch.float32),
                torch.tensor(labels_particle, dtype=torch.float32),
                torch.tensor(label_jet, dtype=torch.float32),
                torch.tensor(self.full_data_array["part_origin"][idx], dtype=torch.float32),
                torch.tensor(self.full_data_array["part_versor"][idx], dtype=torch.float32),
                torch.tensor(self.full_data_array["best_chi2"][idx], dtype=torch.float32))

