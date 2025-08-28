import torch
import lightning as pl
from models.Tiktok.tokenizer import ConvNeXtWithGlobalCompressor, create_convnext_autoencoder, GlobalNerfCompressor
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch.utilities import rank_zero_only
from lightning.fabric.wrappers import _unwrap_objects
from torchvision.utils import make_grid
import wandb

# ---------- lightning module ----------
class LitAutoencoder(pl.LightningModule):
    def __init__(self, model_path, config, device):
        super().__init__()
        self.ae_model = create_convnext_autoencoder(variant = config.model.convnext)
        self.compressor = GlobalNerfCompressor(config.model.hidden_s, config.model.num_proxy, config.model.encoder_layers
                                               , config.model.proxy_layers, config.model.dec_layers)
        self.model = ConvNeXtWithGlobalCompressor(self.ae_model, self.compressor)
        self.lr = config.model.lr
        self.weight_decay = config.model.weight_decay

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch["pixel_values"]
        reconstructed, proxies, commit_loss, indices = self(x)
        loss = F.mse_loss(reconstructed, x) + commit_loss
        self.log("train_loss", loss, prog_bar=True)

        if batch_idx % self.log_interval == 0:
            self.log_images(x, reconstructed, self.global_step)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def log_images(self, x, out, step, n=8):
        # scale [-1,1] -> [0,1]
        x = (x.clamp(-1,1)+1)/2
        out = (out.clamp(-1,1)+1)/2
        grid = make_grid(torch.cat([x[:n], out[:n]], dim=0), nrow=n)
        # log to tensorboard
        self.logger.experiment.log({
            "reconstruction": [wandb.Image(grid, caption=f"step {step}")]
        })

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if batch_idx % 200 == 0:
            x = batch["pixel_values"]
            with torch.no_grad():
                out, _, _, _ = self(x.to(self.device))
            self.save_recon_images(x, out, self.global_step)

