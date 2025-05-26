"""
Example usage of the disentanglement visualization functions.

This script demonstrates how to use the new visualization functions to analyze
the quality of disentangled representations learned by the CRMNIST VAE model.
"""

import torch
import os
from torch.utils.data import DataLoader
from disentanglement_visualization import (
    visualize_disentanglement,
    visualize_latent_interpolation,
    visualize_factor_traversal
)

def run_disentanglement_analysis(model, dataloader, device, output_dir):
    """
    Run comprehensive disentanglement analysis on a trained model.
    
    Args:
        model: Trained VAE model
        dataloader: DataLoader with test/validation data
        device: torch device
        output_dir: Directory to save visualizations
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting disentanglement analysis...")
    
    # 1. Main disentanglement visualization
    print("\n1. Creating disentanglement visualizations...")
    disentanglement_path = os.path.join(output_dir, "disentanglement_analysis.png")
    visualize_disentanglement(
        model=model,
        dataloader=dataloader,
        device=device,
        save_path=disentanglement_path,
        num_variations=7,  # Show 7 variations per latent space
        num_examples=3     # Analyze 3 diverse examples
    )
    
    # 2. Latent space interpolation
    print("\n2. Creating latent interpolation visualizations...")
    interpolation_path = os.path.join(output_dir, "latent_interpolation.png")
    visualize_latent_interpolation(
        model=model,
        dataloader=dataloader,
        device=device,
        save_path=interpolation_path,
        num_steps=7
    )
    
    # 3. Factor traversal analysis
    print("\n3. Creating factor traversal visualizations...")
    traversal_path = os.path.join(output_dir, "factor_traversal.png")
    visualize_factor_traversal(
        model=model,
        device=device,
        save_path=traversal_path,
        num_steps=7
    )
    
    print(f"\nDisentanglement analysis complete! Results saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"- disentanglement_analysis_example_*.png: Individual example analyses")
    print(f"- disentanglement_analysis_summary.png: Summary of all examples")
    print(f"- latent_interpolation.png: Interpolation between different images")
    print(f"- factor_traversal_*.png: Systematic traversal of latent dimensions")


def analyze_disentanglement_quality(model, dataloader, device):
    """
    Provide qualitative analysis guidelines for interpreting the visualizations.
    
    This function doesn't generate visualizations but provides guidance on what
    to look for when evaluating disentanglement quality.
    """
    
    print("\n" + "="*60)
    print("DISENTANGLEMENT QUALITY EVALUATION GUIDE")
    print("="*60)
    
    print("\n1. DISENTANGLEMENT ANALYSIS:")
    print("   Good disentanglement should show:")
    print("   - zy variations: Changes in digit identity while preserving color/rotation")
    print("   - za variations: Changes in rotation/domain while preserving digit/color")
    print("   - zay variations: Interaction effects between digit and domain")
    print("   - zx variations: Style changes without affecting semantic content")
    
    print("\n2. LATENT INTERPOLATION:")
    print("   Good interpolation should show:")
    print("   - Smooth transitions between images")
    print("   - Meaningful intermediate states")
    print("   - Each latent space controlling different factors")
    
    print("\n3. FACTOR TRAVERSAL:")
    print("   Good factor control should show:")
    print("   - Individual dimensions controlling specific factors")
    print("   - Smooth changes as dimension values change")
    print("   - Minimal entanglement between factors")
    
    print("\n4. SIGNS OF POOR DISENTANGLEMENT:")
    print("   - Multiple factors changing when varying one latent space")
    print("   - Abrupt or unrealistic transitions")
    print("   - Similar effects from different latent spaces")
    print("   - Mode collapse or unrealistic generated images")
    
    print("\n5. CRMNIST-SPECIFIC EXPECTATIONS:")
    print("   - zy should primarily control digit identity (0-9)")
    print("   - za should primarily control rotation (0°, 15°, 30°, 45°, 60°, 75°)")
    print("   - Color should be controlled by domain-specific factors")
    print("   - zx should handle residual variations and style")
    
    print("\n" + "="*60)


# Example usage in a training script:
"""
# After training your model, you can run disentanglement analysis like this:

# Load your trained model
model = VAE(...)  # Your trained model
model.load_state_dict(torch.load('path/to/your/model.pth'))
model.eval()

# Get your test dataloader
test_dataloader = DataLoader(...)  # Your test dataset

# Run analysis
run_disentanglement_analysis(
    model=model,
    dataloader=test_dataloader,
    device=device,
    output_dir='./disentanglement_results'
)

# Print evaluation guidelines
analyze_disentanglement_quality(model, test_dataloader, device)
"""

if __name__ == "__main__":
    print("This is an example script showing how to use disentanglement visualization.")
    print("Import the functions and use them in your training/evaluation pipeline.")
    print("\nSee the example usage at the bottom of this file for integration guidance.")
    
    # Print the evaluation guide
    analyze_disentanglement_quality(None, None, None) 