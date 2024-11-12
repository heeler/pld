import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from pathlib import Path
import lightning as L
import torchmetrics
from torchmetrics import Metric
from torchmetrics.functional import accuracy
from tqdm import tqdm


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total',default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct',default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        assert preds.shape == target.shape
        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()


class SimpleClassifier(L.LightningModule):
    def __init__(self, num_inputs: int, num_hidden: int, num_outputs: int):
        super(SimpleClassifier, self).__init__()
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        # self.act_fn = nn.Tanh()
        self.act_fn = nn.ReLU()
        self.linear2 = nn.Linear(num_hidden, num_outputs)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.my_accuracy = MyAccuracy()
        self.accuracy = torchmetrics.Accuracy(task='binary', num_classes=2)
        self.f1_score = torchmetrics.F1Score(task='binary', num_classes=2, average='macro')

    def forward(self, x):
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        return x

    def training_step(self, batch, batch_idx):
        loss, scores, y = self.common_step(batch, batch_idx)
        scores_1d = scores.squeeze(dim=1)
        my_accuracy = self.my_accuracy(scores_1d, y)
        accuracy = self.accuracy(scores_1d, y)
        f1_score = self.f1_score(scores_1d, y)
        self.log_dict({'train_loss': loss, 'train_accuracy': accuracy, 'train_f1_score': f1_score, 'my_accuracy': my_accuracy},
                      on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'scores': scores, 'y': y}

    # def on_train_epoch_end(self, training_step_outputs):
    #     pass

    def validation_step(self, batch, batch_idx):
        loss, scores, y = self.common_step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y = self.common_step(batch, batch_idx)
        return loss

    def common_step(self, batch, batch_idx):
        x, y = batch
        x = x.squeeze(dim=1)
        scores = self.forward(x)
        loss = self.loss_fn(scores.squeeze(dim=1), y.float())
        return loss, scores, y

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.squeeze(dim=1)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


class XORDataset(data.Dataset):
    def __init__(self, size, std=0.1):
        super(XORDataset, self).__init__()
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


def example():
    model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)
    print(model)
    dataset = XORDataset(size=200)
    print(f"Size of dataset: {len(dataset)}")
    print(f"Datapoint 0: {dataset[0]}")
    visualize_samples(dataset.data, dataset.label)
    plt.show()

def training_example():
    print("Device", device)
    model = SimpleClassifier(num_inputs=2, num_hidden=4, num_outputs=1)
    train_dataset = XORDataset(size=1000, std=0.2)
    val_dataset = XORDataset(size=1000)
    train_data_loader = data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    val_data_loader = data.DataLoader(dataset=val_dataset, batch_size=128, shuffle=False)
    model.to(device)

    trainer = L.Trainer(accelerator="gpu",
                        devices=2,
                        strategy='ddp',
                        min_epochs=1,
                        max_epochs=400,
                        precision='bf16-mixed',
                        log_every_n_steps=4)
    trainer.fit(model, train_data_loader, val_data_loader)
    trainer.validate(model, val_data_loader)
    print("visualize js")
    visualize_samples(model, train_dataset.data, train_dataset.label)
    plt.show()
    print("done")

