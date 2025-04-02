import torch

def chi2(origins, versors, weights, fitted_vertex):
    """Compute the chi2 of a set of tracks with respect to a fitted vertex
    Based on the chi2 definition in Pattern recognition, tracking and vertex reconstruction in particle detectors, 2021, R. Frühwirth
    Sec 8.1.1.1 formula 8.1
    chi2 = \sum_i [(o_i - v) x v_i]^2
    NOTE: it misses tracks' errors!!

    Parameters
    ----------
    origins : torch.Tensor (nTracks, 3)
        Origins of the tracks
    versors : torch.Tensor (nTracks, 3)
        Versors of the tracks
    weights : torch.Tensor (nTracks)
        Weights of the tracks, i.e. probability of the track to be from the fitted vertex
    fitted_vertex : torch.Tensor (3)
        Fitted vertex position

    Returns
    -------
    torch.Tensor
        Chi2 of the fit
    """

    # If no tracks' weights are provided, every track has the same weight
    if weights is None:
        weights = torch.ones(origins.shape[0])

    # Computation of the previous formula
    ov = origins - fitted_vertex
    chi2 = torch.cross(ov, versors)
    chi2_norm = torch.linalg.norm(chi2, axis=1) ** 2
    # Multiplication by the probability of the track to be from the fitted vertex
    chi2_weighted = weights * chi2_norm
    # Returning the sum of each track's chi2
    return torch.sum(chi2_weighted)

def fit(origins, versors, weights, fit_iter=100):
    """Fit a vertex to a set of tracks

    Parameters
    ----------
    origins : torch.Tensor (nTracks, 3)
        Origins of the tracks
    versors : torch.Tensor (nTracks, 3)
        Versors of the tracks
    weights : torch.Tensor (nTracks)
        Weights of the tracks, i.e. probability of the track to be from the fitted vertex
    fit_iter : int, optional
        Number of iterations of the fitting, by default 100

    Returns
    -------
    torch.Tensor, torch.Tensor
        Chi2 of the fit, fitted vertex
    """

    # fitted vertex tensor
    fitted_vertex = torch.tensor([0.,0.,0.], requires_grad=True)
    # Adam optimizer to minimize the chi2
    minimizer = torch.optim.Adam([fitted_vertex], lr=0.01)

    # Minimization loop
    for _ in range(fit_iter):
        minimizer.zero_grad()
        loss = chi2(origins, versors, weights, fitted_vertex)
        loss.backward()
        minimizer.step()

    return loss, fitted_vertex
