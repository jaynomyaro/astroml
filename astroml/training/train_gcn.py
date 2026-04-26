import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from astroml.models.gcn import GCN


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Planetoid(root="data", name="Cora", transform=NormalizeFeatures())
    data = dataset[0].to(device)

    model = GCN(
        input_dim=dataset.num_node_features,
        hidden_dim=16,
        output_dim=dataset.num_classes,
        dropout=0.5,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            val_acc = _accuracy(model, data, data.val_mask)
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f}")

    print(f"Test Accuracy: {_accuracy(model, data, data.test_mask):.4f}")


def _accuracy(model: GCN, data, mask) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)
    return float((pred[mask] == data.y[mask]).sum()) / float(mask.sum())


if __name__ == "__main__":
    train()
