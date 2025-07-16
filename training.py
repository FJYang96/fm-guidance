import torch
from tqdm import tqdm
import torchcfm.conditional_flow_matching as cfm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid


def train_fm_model(model, train_loader, n_epochs, sigma, batch_size):
    # Default to using cuda > mps > cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    FM = cfm.ConditionalFlowMatcher(sigma=sigma)
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(n_epochs):
        for i, data in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            x1 = data[0].to(device)
            x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            vt = model(t, xt)
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            optimizer.step()
    
    return model