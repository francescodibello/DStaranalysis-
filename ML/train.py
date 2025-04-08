import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from dataloader import ParticleJetDataset  # Import dataset
from model import ParticleJetClassifier  # Import model
from vertexing import fit

# Load Dataset and Dataloader
root_file = "../data/test.root"  # Change this to your actual ROOT file
dataset = ParticleJetDataset(root_file)
dataloader = DataLoader(dataset, batch_size=100, shuffle=True, collate_fn=lambda x: list(zip(*x)))

# Initialize Model, Loss, and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ParticleJetClassifier().to(device)
criterion_particle = nn.CrossEntropyLoss()
criterion_jet = nn.BCELoss()


# **Define Optimizer**
initial_lr = 5e-4  # Starting learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

# **Adaptive Learning Rate Scheduler**
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# Training Loop
num_epochs = 50
fit_iter = 100

for epoch in range(num_epochs):
    total_loss = 0
    model.train()

    for batch_idx, batch in enumerate(dataloader):  # Add index `batch_idx`
        if batch_idx > 0:  # Stop after processing the first event
            break  

    for batch in dataloader:
        part_features, labels_particle, labels_jet, part_origins, part_versors, best_chi2s = batch
        part_features = [p.to(device) for p in part_features]
        labels_particle = [l.to(device) for l in labels_particle]
        labels_jet = torch.tensor(labels_jet, dtype=torch.float32).to(device)

        optimizer.zero_grad()
        batch_loss = 0

        for particles, lbl_particle, lbl_jet, part_origin, part_versor, best_chi2 in zip(part_features, labels_particle, labels_jet, part_origins, part_versors, best_chi2s):
            particles = particles.unsqueeze(0)
            pred_particle, pred_jet = model(particles)

            loss_particle = criterion_particle(pred_particle.squeeze(0), lbl_particle.to(torch.long))
            loss_jet = criterion_jet(pred_jet.squeeze(0), lbl_jet.view(-1))
            
            # Performing vertex fit
            # Taking the output preds of the D0 class
            weights = pred_particle.squeeze(0)[:,1]
            chi2, _ = fit(part_origin, part_versor, weights, fit_iter)

            loss_vertexing = (chi2 - best_chi2)**2

            total_batch_loss = (loss_particle + loss_jet + loss_vertexing) / len(part_features)
            #total_batch_loss = (loss_particle + loss_jet) / len(part_features)
            batch_loss += total_batch_loss


        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item()
        # **Update Learning Rate Based on Validation Loss**
        #scheduler.step(total_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

print("Training complete!")

