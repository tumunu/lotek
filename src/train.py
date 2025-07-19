"""
LoTek Fractal Network Training Module

Implements the training pipeline for the LoTek fractal cybersecurity neural network,
including optimization, visualization, and performance monitoring.
"""

import wandb
import torch 

from config import BATCH_SIZE, EPOCHS, WANDB_PROJECT
from src.model import FractalNetwork
from src.comparison_model import SimpleTransformer
from src.data import DataProcessor 
from src.visualize import visualize_attention, visualize_embeddings
from src.utils import store_embedding_hash
from src.evaluation import evaluate_models

def train(model, dataloader):
    """Train the LoTek fractal network with visualization and monitoring."""
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.NLLLoss()

    for epoch in range(EPOCHS):
        for batch_idx, batch in enumerate(dataloader):
            X = batch['input_ids']
            Y = X[:, 1:].contiguous().view(-1)  # Flatten Y for consistency with model output
            X = X[:, :-1]

            optimizer.zero_grad()
            output = model(X)  # If 'X' is for the entire sequence, you might need to reshape 'output' for the loss function
            
            # Adjust for the last token prediction if that's what your model does
            # Here, we assume 'output' is for the last token only, so no need to reshape
            loss = criterion(output, Y)
            loss.backward()
            optimizer.step()

            wandb.log({"epoch": epoch, "batch_loss": loss.item()})

            if batch_idx % 1000 == 0:
                # Assuming you have methods to get attention and embeddings from your model
                try:
                    attention_weights, embeddings = model.get_attention_and_embeddings(X)
                    visualize_attention(attention_weights, f'attention_{epoch}_{batch_idx}.png')
                    visualize_embeddings(embeddings, f'embeddings_{epoch}_{batch_idx}.png')
                    wandb.log({"Attention Visualization": wandb.Image(f'attention_{epoch}_{batch_idx}.png')})
                    wandb.log({"Embeddings Visualization": wandb.Image(f'embeddings_{epoch}_{batch_idx}.png')})
                except AttributeError:
                    print("Model does not support attention or embedding visualization")

        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item()}")
        wandb.log({"epoch_loss": loss.item()})

    # Log model artifacts
    artifact = wandb.Artifact('fractal_model', type='model')
    torch.save(model.state_dict(), 'fractal_model.pth')
    artifact.add_file('fractal_model.pth')
    wandb.log_artifact(artifact)

if __name__ == "__main__":
    wandb.init(project=WANDB_PROJECT, config={
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "model_type": "FractalNetwork"
    })
    model = FractalNetwork()
    data_processor = DataProcessor()
    dataloader = data_processor.get_data_loader(BATCH_SIZE)
    train(model, dataloader)