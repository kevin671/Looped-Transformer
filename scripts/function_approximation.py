# %%
import torch
from .models.looped_transformer import LoopedTransformer, LoopedConfig


def generate_func(data_points, seq_len, dim):
    # random noise

    x = torch.randn(data_points, seq_len, dim)
    y = torch.randn(data_points, seq_len, dim)
    return x, y


def train(x, model, epochs=100, batch_size=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    l1 = torch.nn.L1Loss()
    for epoch in range(epochs):
        for i in range(0, x.shape[0], batch_size):
            x_batch = x[i : i + batch_size]
            y_batch = x[i : i + batch_size]
            optimizer.zero_grad()
            y_hat = model(x_batch)
            loss = l1(y_hat, y_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} Loss {loss.item()}")
    return model


def test(y, model):
    l1 = torch.nn.L1Loss()
    y_hat = model(y)
    loss = l1(y_hat, y)
    print(f"Test Loss {loss.item()}")
    return y_hat


if __name__ == "__main__":

    class Model(torch.nn.Module):
        def __init__(self, seq_len=5, dim=2):
            super(Model, self).__init__()
            self.linear = torch.nn.Linear(dim, dim)
            self.lstm = torch.nn.LSTM(dim, dim, batch_first=True)

        def forward(self, x):
            x, _ = self.lstm(x)
            x = self.linear(x)
            return x

    x, y = generate_func(10, 5, 2)
    print(x.shape)
    model = Model()
    config = LoopedConfig(
        n_layer=1, n_head=2, hidden_dim=24, bias=True, use_time_embed=False
    )

    # model = LoopedTransformer(config)
    # model = model.cuda()

    train(x, model)
    test(y, model)

# %%
