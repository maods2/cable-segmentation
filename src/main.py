import argparse
from dataset import ImageMaskDataset
from config import build_transforms, load_config, get_dataloaders
from train import train_model


def main():
    # argparse configuration
    parser = argparse.ArgumentParser(
        description="Train model with specified configuration."
    )
    parser.add_argument(
        "--train", action="store_true", help="Flag to initiate training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()

    # Load the configuration file
    config = load_config(args.config)
    dataset = ImageMaskDataset(
        image_dir=config.data_path, transform=build_transforms(config)
    )

    # Start training if the `--train` flag is set
    if args.train:
        train_loader, val_loader, test_loader = get_dataloaders(config, dataset)
        print(f"train size: {len(train_loader.dataset)}, num batches: {len(train_loader)}")
        print(f"val size: {len(val_loader.dataset)}")
        print(f"test size: {len(test_loader.dataset)}")
        train_model(config, train_loader, val_loader, test_loader)
    else:
        print("Train flag not set. Use --train to start training.")


if __name__ == "__main__":
    main()
