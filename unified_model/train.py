import os
import argparse
import torch

from unified_model.data.mnist_data import get_mnist_dataloader
from unified_model.models.unet import UNet, ConditionalUNet
from unified_model.models.diffusion_model import DiffusionModel, ConditionalDiffusionModel
from unified_model.trainers.diffusion_trainer import DiffusionTrainer

def parse_args():
    """Parse command line arguments for training"""
    parser = argparse.ArgumentParser(description='Training script for generative models')
    
    # General arguments
    parser.add_argument('--model', type=str, default='diffusion', 
                        choices=['diffusion', 'conditional_diffusion'],
                        help='Model type to use')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--device', type=str, default=None, 
                        help='Device to use (default: auto-detect)')
    parser.add_argument('--output_dir', type=str, default='out', help='Output directory')
    parser.add_argument('--model_path', type=str, default=None, help='Path to load model from')
    parser.add_argument('--save_interval', type=int, default=10, help='Interval to save model checkpoints')
    
    # Diffusion model arguments
    parser.add_argument('--sampler', type=str, default='ddpm', choices=['ddpm'],
                        help='Sampler type for diffusion model')
    parser.add_argument('--n_steps', type=int, default=1000, help='Number of diffusion steps')
    
    # Conditional diffusion arguments
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes (for conditional models)')
    parser.add_argument('--label_emb_dim', type=int, default=32, help='Size of label embedding (for conditional models)')
    
    return parser.parse_args()

def main():
    """Main training function"""
    args = parse_args()
    
    # Auto-detect device if not specified
    if args.device is None:
        args.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {args.device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model based on type
    if args.model == 'diffusion':
        # Initialize diffusion model
        print("Initializing Diffusion model...")
        unet = UNet(args.n_steps, device=args.device)
        model = DiffusionModel(
            unet, 
            sampler_type=args.sampler, 
            n_steps=args.n_steps, 
            device=args.device
        )
        trainer = DiffusionTrainer(
            model, 
            args.device, 
            os.path.join(args.output_dir, args.model)
        )
    
    elif args.model == 'conditional_diffusion':
        # Initialize conditional diffusion model
        print("Initializing Conditional Diffusion model...")
        conditional_unet = ConditionalUNet(
            args.n_steps, 
            num_classes=args.num_classes,
            label_emb_dim=args.label_emb_dim,
            device=args.device
        )
        model = ConditionalDiffusionModel(
            conditional_unet, 
            num_classes=args.num_classes,
            sampler_type=args.sampler, 
            n_steps=args.n_steps, 
            device=args.device
        )
        trainer = DiffusionTrainer(
            model, 
            args.device, 
            os.path.join(args.output_dir, args.model)
        )
    
    # Load model if path is provided
    if args.model_path:
        print(f"Loading model from {args.model_path}...")
        trainer.load_model(args.model_path)
    
    # Training
    print(f"Training {args.model} model...")
    train_loader = get_mnist_dataloader(
        args.batch_size, 
        train=True
    )
    
    trainer.train(
        train_loader, 
        n_epochs=args.n_epochs,
        save_interval=args.save_interval
    )

if __name__ == '__main__':
    main() 
