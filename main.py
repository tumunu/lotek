"""
LoTek Fractal Cybersecurity Neural Network - Main Application Entry Point

This module serves as the primary entry point for the LoTek fractal cybersecurity neural network
AI system. It initializes the fractal model, handles training workflows,
and manages the integration between neural network components and cybersecurity modules.
"""

import os
import sys
import torch
from src.fractal_modules.unification import FractalUnifyingModel
from src.data import DataProcessor
from src.train import train
from src.evaluation import evaluate_models
from src.comparison_model import SimpleTransformer
from transformers import AutoTokenizer
import wandb
from config import BATCH_SIZE, EPOCHS, WANDB_PROJECT

if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main training and evaluation pipeline for the LoTek fractal cybersecurity neural network."""
    wandb.init(project=WANDB_PROJECT, config={
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "model_type": "FractalNetwork"
    })

    fractal_model = FractalUnifyingModel(vocab_size=50257, embedding_dim=768)
    comparison_model = SimpleTransformer()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    data_processor = DataProcessor()
    dataloader = data_processor.get_data_loader(batch_size=BATCH_SIZE)

    print("Training the LoTek fractal cybersecurity neural network...")
    train(fractal_model, dataloader)

    print("Evaluating models...")
    eval_dataloader = data_processor.get_data_loader(batch_size=8)
    results = evaluate_models(fractal_model, comparison_model, tokenizer, eval_dataloader)
    
    print("Evaluation Results:")
    for model, metrics in results.items():
        print(f"{model}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

    print("Performing inference with the LoTek fractal network...")
    test_input = tokenizer.encode("The quick brown fox jumps over the", return_tensors='pt')
    with torch.no_grad():
        output = fractal_model(test_input)
    print("Fractal Model Output (log probabilities):", output)

    for model_name, metrics in results.items():
        wandb.run.summary[f"{model_name}_metrics"] = metrics

    torch.save(fractal_model.state_dict(), 'fractal_model_weights.pth')
    artifact = wandb.Artifact('fractal_model', type='model')
    artifact.add_file('fractal_model_weights.pth')
    wandb.log_artifact(artifact)

if __name__ == "__main__":
    main()