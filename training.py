import torch
import torchcfm.conditional_flow_matching as cfm


def train_fm_model(
    model, train_loader, n_epochs, sigma, val_loader=None, conditioned=False
):
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
        for i, data in enumerate(train_loader):
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
            train_loss = torch.mean((vt - ut) ** 2)
            train_loss.backward()
            optimizer.step()
        # print(f"Epoch {epoch:02d} Training loss: {train_loss.item():.4f}")

        if val_loader is not None and epoch % 2 == 0:
            with torch.no_grad():
                val_loss = 0.0
                for i, data in enumerate(val_loader):
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
                    val_loss += torch.mean((vt - ut) ** 2).item()
                print(
                    f"Epoch {epoch:02d} Validation loss: {val_loss / len(val_loader):.4f}"
                )

    torch.save(model.state_dict(), f"models/model_{epoch}.pth")
    model.to("cpu").eval()
    return model
