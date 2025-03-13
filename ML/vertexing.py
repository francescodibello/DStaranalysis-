import torch

def chi2(origins, versors, weights, fitted_vertex):
    if weights is None:
        weights = torch.ones(origins.shape[0])

    ov = origins - fitted_vertex
    chi2 = torch.cross(ov, versors)
    chi2_norm = torch.linalg.norm(chi2, axis=1) ** 2
    chi2_weighted = weights * chi2_norm
    return torch.sum(chi2_weighted)

def fit(origins, versors, weights, fit_iter=100):
    fitted_vertex = torch.tensor([0.,0.,0.], requires_grad=True)
    minimizer = torch.optim.Adam([fitted_vertex], lr=0.01)

    for _ in range(fit_iter):
        minimizer.zero_grad()
        loss = chi2(origins, versors, weights, fitted_vertex)
        loss.backward()
        minimizer.step()
    return loss, fitted_vertex
