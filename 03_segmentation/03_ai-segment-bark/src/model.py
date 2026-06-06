import segmentation_models_pytorch as smp
import torch


def create_model(config: dict) -> torch.nn.Module:
    """Create segmentation model from config using segmentation_models_pytorch."""
    model_cfg = config["model"]
    num_classes = config["dataset"]["num_classes"]

    architecture = model_cfg["architecture"]
    encoder = model_cfg["encoder"]
    encoder_weights = model_cfg["encoder_weights"]

    model_factory = {
        "DeepLabV3Plus": smp.DeepLabV3Plus,
        "DeepLabV3": smp.DeepLabV3,
        "Unet": smp.Unet,
        "UnetPlusPlus": smp.UnetPlusPlus,
        "FPN": smp.FPN,
        "PSPNet": smp.PSPNet,
        "MAnet": smp.MAnet,
    }

    if architecture not in model_factory:
        raise ValueError(f"Unknown architecture: {architecture}. Choose from {list(model_factory.keys())}")

    model = model_factory[architecture](
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
    )

    return model


def count_parameters(model: torch.nn.Module) -> dict:
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}
