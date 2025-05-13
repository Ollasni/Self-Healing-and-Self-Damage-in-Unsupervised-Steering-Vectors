import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_head_results(results_tensor: torch.Tensor, title: str = "Head Results", cmap: str = "viridis", figsize: tuple = (6, 7), 
                      vmin: float = None, vmax: float = None, colorbar_label: str = "Value") -> tuple:
    """
    Plot a heatmap of head results.
    
    Args:
        results_tensor: Tensor of shape [n_layers, n_heads] containing results to plot
        title: Title for the plot
        cmap: Colormap to use
        figsize: Figure size as (width, height)
        vmin, vmax: Min and max values for color scaling
        colorbar_label: Label for the colorbar
    """

    
    # Convert tensor to numpy if needed
    if isinstance(results_tensor, torch.Tensor):
        results_tensor = results_tensor.cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(results_tensor, cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)
    
    # Set labels and title
    ax.set_xlabel('Head')
    ax.set_ylabel('Layer')
    ax.set_title(title)
    
    # Set integer ticks for layers and heads
    n_layers, n_heads = results_tensor.shape
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_layers))
    
    plt.tight_layout()
    return fig, ax