"""
LoTek Fractal Network Visualization Module

Provides visualization capabilities for the LoTek fractal cybersecurity neural network,
including attention weight heatmaps and embedding scatter plots.
"""

import matplotlib.pyplot as plt
import torch

def visualize_attention(attention_weights, save_path='attention_visualization.png'):
    """Visualize LoTek fractal attention weights as a heatmap."""
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(attention_weights[0].detach().cpu().numpy(), cmap='hot', interpolation='nearest')
    ax.set_title('LoTek Fractal Attention Weights')
    plt.colorbar(im)
    plt.savefig(save_path)
    plt.close(fig)

def visualize_embeddings(embeddings, save_path='token_embeddings.png'):
    """Visualize LoTek fractal network token embeddings as a scatter plot."""
    embeddings = embeddings.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(embeddings[:, 0], embeddings[:, 1], c=range(embeddings.shape[0]), cmap='viridis')
    ax.set_title('LoTek Fractal Network Token Embeddings')
    plt.colorbar(ax.scatter([], [], c=[]), ax=ax, label='Token Index')
    plt.savefig(save_path)
    plt.close(fig)