from typing import Any
import lightning as pl
import torch
import torch.nn as nn
from architecture.full_precision.models.snn.l2mu import L2MU


class NetworkEngine(pl.LightningModule):
    def __init__(
            self,
            num_inputs,
            num_outputs,
            architecture,
            params,
    ):
        super().__init__()

        self.architecture = architecture
        self.model = L2MU(input_size=num_inputs, output_size=num_outputs, params=params, neuron_type='Leaky')
        self.loss_fn = nn.CrossEntropyLoss()
        self.running_length = 0
        self.running_total = 0
        self.lr = params["lr"]
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, betas=(0.9, 0.999)
        )
        return optimizer

    def forward(self, input) -> Any:

        input = input.swapaxes(1, 0)
        return self.model(input)

    def training_step(self, batch, batch_idx):
        train_data, train_labels = batch
        train_data = train_data.swapaxes(1, 0)
        pred_output = self.model(train_data)

        # measure accuracy
        batch_accuracy = self.calc_accuracy(pred_output, train_labels)
        self.log("train_accuracy", batch_accuracy, prog_bar=True)

        # measure loss
        train_loss = self.loss_fn(pred_output.sum(0), train_labels)
        self.log("train_loss", train_loss, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        validation_data, validation_labels = batch
        validation_data = validation_data.swapaxes(1, 0)

        # Val set forward pass
        pred_output = self.model(validation_data)

        # measure accuracy
        batch_accuracy = self.calc_accuracy(pred_output, validation_labels)
        self.log("val_accuracy", batch_accuracy, prog_bar=True)

        # measure loss
        val_loss = self.loss_fn(pred_output.sum(0), validation_labels)
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        test_data, test_labels = batch
        test_data = test_data.swapaxes(1, 0)

        pred_output = self.model(test_data)

        # measure accuracy
        batch_accuracy = self.calc_accuracy(pred_output, test_labels)
        self.log("test_accuracy", batch_accuracy, prog_bar=True)

        # measure loss
        test_loss = self.loss_fn(pred_output.sum(0), test_labels)
        self.log("test_loss", test_loss, prog_bar=True)
        return test_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        test_data, test_labels = batch
        test_data = test_data.swapaxes(1, 0)

        pred_output = self.model(test_data)

        _, pred = pred_output.sum(dim=0).max(1)

        return pred.detach().cpu().numpy()

    @staticmethod
    def calc_accuracy(output, labels):
        _, idx = output.sum(dim=0).max(1)
        label_count = (labels == idx).sum()
        return label_count / len(labels)