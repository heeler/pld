import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import to_rgba

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class SimpleClassifier(nn.Module):
    def __init__(self, num_inputs: int, num_hidden: int, num_outputs: int):
        super(SimpleClassifier, self).__init__()
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.act_fn = nn.Tanh()
        self.linear2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        return x

class XORDataset(data.Dataset):
    def __init__(self, size, std=0.1):
        super().__init__()
        self.size = size
        self.std = std
        self.generate_continuous_xor()

    def generate_continuous_xor(self):
        data = torch.randint(low=0, high=2, size=(self.size, 2), dtype=torch.float32)
        label = (data.sum(dim=1) == 1).to(torch.long)
        data += self.std * torch.randn(data.shape)

        self.data = data
        self.label = label

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        data_point = self.data[index]
        data_label = self.label[index]
        return data_point, data_label

@torch.no_grad()
def visualize_samples(model, data, label):
    if isinstance(data, torch.Tensor):
        data=data.cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()
    data_0= data[label == 0]
    data_1= data[label == 1]

    plt.figure(figsize=(4, 4))
    plt.scatter(data_0[:,0], data_0[:,1], edgecolors='#333', label='Class 0')
    plt.scatter(data_1[:,0], data_1[:,1], edgecolors='#333', label='Class 1')
    plt.title("Dataset samples")
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.legend()

    model.to(device)
    c0 = torch.Tensor(to_rgba("C0")).to(device)
    c1 = torch.Tensor(to_rgba("C1")).to(device)
    x1 = torch.arange(-0.5, 1.5, step=0.01, device=device)
    x2 = torch.arange(-0.5, 1.5, step=0.01, device=device)
    xx1, xx2 = torch.meshgrid(x1, x2)  # Meshgrid function as in numpy
    model_inputs = torch.stack([xx1, xx2], dim=-1)
    preds = model(model_inputs)
    preds = torch.sigmoid(preds)
    # Specifying "None" in a dimension creates a new one
    output_image = (1 - preds) * c0[None, None] + preds * c1[None, None]
    output_image = (
        output_image.cpu().numpy()
    )  # Convert to numpy array. This only works for tensors on CPU, hence first push to CPU
    plt.imshow(output_image, origin="lower", extent=(-0.5, 1.5, -0.5, 1.5))
    plt.grid(False)

def train_model(model, optimizer, data_loader, loss_module, num_epochs=100):
    # Set the model to training mode!
    model.train()

    #training loop
    for epoch in range(num_epochs):
        for data_inputs, data_labels in data_loader:

            ## Step 1: Move input data to the device (only necessary for GPU use)
            data_inputs, data_labels = data_inputs.to(device), data_labels.to(device)

            ## Step 2: Run the model on the input data
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1) # Output is [Batch size, 1] but we want [Batch size]

            ## Step 3: Calculate theloss
            loss = loss_module(preds, data_labels.float())

            ## Step 4: Perform backprop
            # before calculating the gradients, we need to ensure the gradients are initialized to zero
            # otherwise the newly calculated gradients would be added to the prior values
            optimizer.zero_grad()
            # perform backprop
            loss.backward()

            ## Step 5:Update the model parameters
            optimizer.step()



def example():
    model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)
    print(model)
    dataset = XORDataset(size=200)
    print(f"Size of dataset: {len(dataset)}")
    print(f"Datapoint 0: {dataset[0]}")
    visualize_samples(model, dataset.data, dataset.label)
    plt.show()

def training_example():
    print("Device", device)
    model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)
    train_dataset = XORDataset(size=1000)
    train_data_loader = data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    model.to(device)
    loss_module = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    train_model(model, optimizer, data_loader=train_data_loader, loss_module=loss_module, num_epochs=100)
    state_dict = model.state_dict()
    print(state_dict)
    visualize_samples(model, train_dataset.data, train_dataset.label)
    print("show plot")
    plt.show()
    model_dir = Path("~/Sandbox/Python/pld/Models").expanduser()
    torch.save(state_dict, model_dir / "model.tar")

if __name__ == "__main__":
    training_example()
    # example()
