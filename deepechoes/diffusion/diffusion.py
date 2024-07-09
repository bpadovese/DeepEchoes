import tables
import argparse
from noise_schedulers import NoiseScheduler
from dataset import HDF5Dataset, line_dataset, dino_dataset
from torch.utils.data import DataLoader
from torch import nn
from positional_embeddings import PositionalEmbedding
from pathlib import Path
import torch
import numpy as np

class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))
    
class MLP(nn.Module):
    def __init__(self, hidden_size: int = 128, hidden_layers: int = 3, emb_size: int = 128,
                 time_emb: str = "sinusoidal", input_emb: str = "sinusoidal"):
        super().__init__()

        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0)

        concat_size = len(self.time_mlp.layer) + len(self.input_mlp1.layer) + len(self.input_mlp2.layer)
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, 2))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        x1_emb = self.input_mlp1(x[:, 0])
        x2_emb = self.input_mlp2(x[:, 1])
        t_emb = self.time_mlp(t)
        x = torch.cat((x1_emb, x2_emb, t_emb), dim=-1)
        x = self.joint_mlp(x)
        return x

def main(dataset, train_table="/train", train_batch_size=32, **kwargs):
    db = tables.open_file(dataset, mode='r')
    table = db.get_node(train_table + '/data')
    dataset = HDF5Dataset(table, transform=None)
    print(dataset)
    dataloader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)
    # Example of iterating through the DataLoader
    for batch in dataloader:
        print(batch.shape)
        exit()
    db.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument('--train_table', default='/train', type=str)
    parser.add_argument("--output_folder", type=str, default=None)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "quadratic"])
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--time_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "zero"])
    parser.add_argument("--input_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "identity"])
    parser.add_argument("--save_images_step", type=int, default=1)
    args = parser.parse_args()

   

    hidden_size = args.hidden_size
    hidden_layers = args.hidden_layers
    embedding_size = args. embedding_size
    time_embedding = args. time_embedding
    input_embedding = args.input_embedding
    num_timesteps = args.num_timesteps
    beta_schedule = args.beta_schedule
    learning_rate = args.learning_rate
    output_folder = args.output_folder

    dataset = dino_dataset()
    data_loader = DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
    
    model = MLP(
        hidden_size=hidden_size,
        hidden_layers=hidden_layers,
        emb_size=embedding_size,
        time_emb=time_embedding,
        input_emb=input_embedding)

    noise_scheduler = NoiseScheduler(
        num_timesteps=num_timesteps,
        beta_schedule=beta_schedule)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    if output_folder is None:
        output_folder = Path('.').resolve()
    else:
        output_folder = Path(output_folder).resolve()
    
    output_folder.parent.mkdir(parents=True, exist_ok=True)

    # Training loop
    global_step = 0
    losses = []
    model.train()
    for epoch in range(args.num_epochs):
        for batch_idx, batch in enumerate(data_loader):
            batch = batch[0]
            noise = torch.randn(batch.shape) # Creating a noise tensor (Gaussian distribution) with the same shape as the batch
            # For each item in the batch, this selects a random timestep from 0 to num_timesteps - 1 when noise will be added.
            timesteps = torch.randint(0, noise_scheduler.num_timesteps, (batch.shape[0],)).long()
            noisy = noise_scheduler.add_noise(batch, noise, timesteps)
            
            noise_pred = model(noisy, timesteps)
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clipping to prevent the gradients from exploding
            optimizer.step()

            logs = {"loss": loss.detach().item(), "step": global_step}
            losses.append(loss.detach().item())

            global_step += 1

        if epoch % args.save_images_step == 0 or epoch == args.num_epochs - 1:
            # Generate and save images or data
            # Here you would have code that generates data/images based on the model's current state
            # For example, use the model to generate samples and save them
            model.eval()
            sample = torch.randn(args.eval_batch_size, 2) # Drawing from Gaussian distribution
            timesteps = list(range(len(noise_scheduler)))[::-1]
            generated_data = model.generate_sample()  # Assuming generate_sample is a method of MLP
            np.save(output_folder / f"generated_data_epoch_{epoch}_batch_{batch_idx}.npy", generated_data)
            model.train()
    
        print(f"Epoch {epoch} completed, loss: {loss.item()}")
    
    torch.save(model.state_dict(), output_folder / f"dino.pth")
    print("Training completed.")

    exit()
    main(**vars(args))
    
