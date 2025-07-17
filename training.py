import torch
import torchcfm.conditional_flow_matching as cfm
from tqdm import tqdm


def train_fm_model(model, train_loader, n_epochs, sigma, conditioned=False):
    # Default to using cuda > mps > cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    FM = cfm.ConditionalFlowMatcher(sigma=sigma)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(n_epochs):
        for i, data in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            if conditioned:
                x1, y = data[0].to(device), data[1].to(device)
            else:
                x1 = data[0].to(device)
            x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            if conditioned:
                vt = model(t, xt, y)
            else:
                vt = model(t, xt)
            loss = torch.mean((vt - ut) ** 2)
            print(f"Trainingloss: {loss.item():.4f}")
            loss.backward()
            optimizer.step()

    return model
