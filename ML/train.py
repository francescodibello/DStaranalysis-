import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import ParticleJetDataset  # Import dataset
from model import ParticleJetClassifier  # Import model

# Load Dataset and Dataloader
root_file = "test.root"  # Change this to your actual ROOT file
dataset = ParticleJetDataset(root_file)
dataloader = DataLoader(dataset, batch_size=100, shuffle=True, collate_fn=lambda x: list(zip(*x)))

# Initialize Model, Loss, and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ParticleJetClassifier().to(device)
criterion_particle = nn.BCELoss(reduction="mean")
criterion_jet = nn.BCELoss(reduction="mean")


optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
num_epochs = 50
for epoch in range(num_epochs):
    total_loss = 0
    model.train()

    for batch_idx, batch in enumerate(dataloader):  # Add index `batch_idx`
        if batch_idx > 0:  # Stop after processing the first event
            break  

    for batch in dataloader:
        part_features, labels_particle, labels_jet = batch
        part_features = [p.to(device) for p in part_features]
        labels_particle = [l.to(device) for l in labels_particle]
        labels_jet = torch.tensor(labels_jet, dtype=torch.float32).to(device)

        optimizer.zero_grad()
        batch_loss = 0

        for particles, lbl_particle, lbl_jet in zip(part_features, labels_particle, labels_jet):
            particles = particles.unsqueeze(0)
            pred_particle, pred_jet = model(particles)


            loss_particle = criterion_particle(pred_particle.squeeze(0), lbl_particle)
            loss_jet = criterion_jet(pred_jet.squeeze(0), lbl_jet.view(-1))

            total_batch_loss = (loss_particle + loss_jet) / len(part_features)
            batch_loss += total_batch_loss

        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

print("Training complete!")

