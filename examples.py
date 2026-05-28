"""
Example usage of the Steering Angle Prediction package.
Demonstrates training, evaluation, and inference.
"""

import torch
import torch.nn as nn
from pathlib import Path

from config.config import (
    DATA_DIR, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE,
    INPUT_CHANNELS, HIDDEN_SIZE, OUTPUT_SIZE
)
from src.data import create_dataloaders, get_transforms
from src.models import create_model
from src.training import Trainer, create_optimizer, create_scheduler
from src.evaluation import evaluate_model
from src.utils import (
    setup_logging, get_device, seed_everything, count_parameters
)


def example_training():
    """
    Example: Train the model from scratch.
    """
    print("\n" + "="*60)
    print("EXAMPLE: Training Model")
    print("="*60)
    
    setup_logging()
    seed_everything(42)
    device = get_device("cuda", verbose=True)
    
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=str(DATA_DIR),
        batch_size=BATCH_SIZE
    )
    print(f"✓ Train samples: {len(train_loader.dataset)}")
    print(f"✓ Val samples: {len(val_loader.dataset)}")
    print(f"✓ Test samples: {len(test_loader.dataset)}")
    
    print("\nCreating model...")
    model = create_model(
        input_channels=INPUT_CHANNELS,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        device=device
    )
    print(f"✓ Model parameters: {count_parameters(model):,}")
    
    print("\nSetting up training...")
    criterion = nn.MSELoss()
    optimizer = create_optimizer(model, learning_rate=LEARNING_RATE)
    scheduler = create_scheduler(optimizer)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler
    )
    
    print(f"\nTraining for {NUM_EPOCHS} epochs...")
    history = trainer.train(num_epochs=NUM_EPOCHS)
    
    print("\nSaving model...")
    trainer.save_checkpoint("models/steering_model.pth")
    print("✓ Model saved successfully!")
    
    return model, trainer


def example_evaluation():
    """
    Example: Evaluate a trained model.
    """
    print("\n" + "="*60)
    print("EXAMPLE: Evaluating Model")
    print("="*60)
    
    setup_logging()
    device = get_device("cuda", verbose=True)
    
    print("\nLoading model...")
    model = create_model(
        input_channels=INPUT_CHANNELS,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        device=device
    )
    
    checkpoint_path = "models/steering_model.pth"
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded from {checkpoint_path}")
    else:
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return
    
    print("\nLoading test data...")
    _, _, test_loader = create_dataloaders(
        data_dir=str(DATA_DIR),
        batch_size=BATCH_SIZE
    )
    print(f"✓ Test samples: {len(test_loader.dataset)}")
    
    print("\nEvaluating model...")
    criterion = nn.MSELoss()
    metrics = evaluate_model(model, test_loader, criterion, device)
    
    print("\nResults:")
    for metric, value in metrics.items():
        print(f"  {metric.upper()}: {value:.6f}")


def example_inference():
    """
    Example: Run inference on a single image.
    """
    print("\n" + "="*60)
    print("EXAMPLE: Inference on Single Image")
    print("="*60)
    
    from PIL import Image
    
    setup_logging()
    device = get_device("cuda", verbose=True)
    
    print("\nCreating model...")
    model = create_model(
        input_channels=INPUT_CHANNELS,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        device=device
    )
    model.eval()
    
    checkpoint_path = "models/steering_model.pth"
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded from {checkpoint_path}")
    else:
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return
    
    print("\nLoading test image...")
    test_image_path = "Test_Images/image_001.jpg"
    
    if not Path(test_image_path).exists():
        print(f"✗ Test image not found: {test_image_path}")
        print("  Please ensure a test image exists at the path above")
        return
    
    image = Image.open(test_image_path).convert('RGB')
    print(f"✓ Image loaded: {image.size}")
    
    print("\nPreparing image for inference...")
    transform = get_transforms(is_training=False)
    image_tensor = transform(image).unsqueeze(0).to(device)
    print(f"✓ Image tensor shape: {image_tensor.shape}")
    
    print("\nRunning inference...")
    with torch.no_grad():
        output = model(image_tensor)
        steering_angle = output.item()
    
    print(f"\n✓ Predicted Steering Angle: {steering_angle:.6f} radians")
    print(f"  ({steering_angle * 180 / 3.14159:.2f} degrees)")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Steering Angle Prediction - Example Usage")
    print("="*60)
    
    print("\nAvailable examples:")
    print("  1. Training")
    print("  2. Evaluation")
    print("  3. Inference")
    print("  4. All examples")
    
    choice = input("\nSelect example (1-4): ").strip()
    
    try:
        if choice == "1":
            example_training()
        elif choice == "2":
            example_evaluation()
        elif choice == "3":
            example_inference()
        elif choice == "4":
            example_training()
            example_evaluation()
            example_inference()
        else:
            print("Invalid choice!")
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()
