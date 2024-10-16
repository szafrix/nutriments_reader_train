import wandb
import torch
import lightning as L
import torchmetrics
from config import LOSS, OPTIMIZER

torch.set_float32_matmul_precision("high")


class L_ImageEncoder(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = LOSS
        self.optimizer = OPTIMIZER["optimizer"](
            self.model.parameters(), **OPTIMIZER["optimizer_params"]
        )
        self.initialize_metrics()

    def forward(self, x):
        return self.model(x)

    def _log_metrics(self, y_hat, y, metrics, prefix):
        on_step = prefix == "train"
        for name, metric_collection in metrics.items():
            if name == "general":
                metric_collection(y_hat, y)
                self.log_dict(metric_collection, on_step=on_step, on_epoch=True)
            else:
                nutrient_index = ["kcal", "protein", "carbs", "fat"].index(name)
                metric_collection(y_hat[:, nutrient_index], y[:, nutrient_index])
                self.log_dict(metric_collection, on_step=on_step, on_epoch=True)
        self.log(f"{prefix}_loss", self.loss(y_hat, y), on_step=on_step, on_epoch=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        self._log_metrics(y_hat, y, self.train_metrics, "train")
        return self.loss(y_hat, y)

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        self._log_metrics(y_hat, y, self.val_metrics, "val")

        self.validation_step_outputs.append(
            {"y_hat": y_hat.detach(), "y": y.detach(), "x": x}
        )
        return self.loss(y_hat, y)

    def on_validation_epoch_end(self):
        all_y_hat = torch.cat(
            [output["y_hat"] for output in self.validation_step_outputs]
        )
        all_y = torch.cat([output["y"] for output in self.validation_step_outputs])
        all_x = torch.cat([output["x"] for output in self.validation_step_outputs])

        self.log_images_for_3_best_and_worst_predictions(all_y_hat, all_y, all_x)

        self.validation_step_outputs.clear()

    def log_images_for_3_best_and_worst_predictions(self, y_hat, y, x):
        errors = torch.abs(y_hat - y)

        best_indices = torch.argsort(errors.sum(dim=1))[:3]
        worst_indices = torch.argsort(errors.sum(dim=1), descending=True)[:3]

        for i, idx in enumerate(best_indices):
            img = x[idx]
            self.logger.experiment.log({f"best_prediction_{i+1}": wandb.Image(img)})

        for i, idx in enumerate(worst_indices):
            img = x[idx]
            self.logger.experiment.log({f"worst_prediction_{i+1}": wandb.Image(img)})

    def configure_optimizers(self):
        return self.optimizer

    @staticmethod
    def create_metric_collection(prefix):
        return torchmetrics.MetricCollection(
            {
                "mse": torchmetrics.regression.MeanSquaredError(),
                "mae": torchmetrics.regression.MeanAbsoluteError(),
            },
            prefix=prefix,
        )

    def initialize_metrics(self):
        metric_types = ["general", "kcal", "protein", "carbs", "fat"]
        prefixes = ["train_", "val_"]

        for prefix in prefixes:
            setattr(self, f"{prefix[:-1]}_metrics", {})
            for metric_type in metric_types:
                metric_name = (
                    f"{prefix}{metric_type}_" if metric_type != "general" else prefix
                )
                metric = self.create_metric_collection(metric_name)
                setattr(self, f"{prefix}{metric_type}_metrics", metric)
                getattr(self, f"{prefix[:-1]}_metrics")[metric_type] = metric
