import torch

from config.settings import settings
from inference.types import SegmentationResult


class SegmentationModel:
    """Runs inference and produces a class probability map and a vertebra ID mask."""

    _NUM_VERTEBRAE = 17
    _FG_THRESHOLD = 0.5
    _LOG_EPSILON = 1e-6

    def __init__(self, model: torch.nn.Module, sigma: float = 0.03):
        """Wrap a segmentation model for inference."""
        self.model = model
        self.sigma = sigma
        self.vertebra_centers = torch.linspace(0.05, 0.95, self._NUM_VERTEBRAE)

    def __call__(self, img_tensor: torch.Tensor) -> SegmentationResult:
        """Return probs (C, H, W) and id_mask (H, W) for the given image tensor."""
        probs = self._forward(img_tensor)
        fg_mask = probs[1:].sum(dim=0) > self._FG_THRESHOLD
        id_mask = (torch.argmax(probs[1:], dim=0) + 1) * fg_mask
        return SegmentationResult(id_mask.cpu().numpy(), probs.cpu().numpy())

    def _forward(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return per-class softmax probabilities."""
        with torch.no_grad(), torch.amp.autocast(settings.device):
            batch_img_tensor = img_tensor.unsqueeze(0).to(settings.device)
            outputs = self.model(batch_img_tensor)
        z = outputs["z"]
        fg_logits = outputs["fg_logits"]
        refine_logits = outputs["refine_logits"]
        probs = self._build_probs(z, fg_logits)
        final_logits = refine_logits + torch.log(probs + self._LOG_EPSILON)
        return torch.softmax(final_logits, dim=1)[0]

    def _build_probs(self, z: torch.Tensor, fg_logits: torch.Tensor) -> torch.Tensor:
        """Combine foreground probability and soft vertebra assignment into a prior."""
        fg = torch.sigmoid(fg_logits)
        vertebra_probs = self._soft_assign(z) * fg
        bg = 1 - fg
        return torch.cat([bg, vertebra_probs], dim=1)

    def _soft_assign(self, z: torch.Tensor) -> torch.Tensor:
        """Assign each pixel to a vertebra center using Gaussian similarity."""
        centers = self.vertebra_centers.to(z.device)
        z_expanded = z - centers.view(1, -1, 1, 1)
        logits = -(z_expanded**2) / self.sigma
        return torch.softmax(logits, dim=1)
