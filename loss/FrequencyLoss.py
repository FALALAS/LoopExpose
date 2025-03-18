import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F


class FFT_Loss(nn.Module):
    def __init__(self, reduction='mean'):
        """
        FFT Loss class that computes the Mean Absolute Error (MAE) loss in the frequency domain.

        Args:
            reduction (str): Reduction method for the loss ('mean' or 'sum'). Default is 'mean'.
        """
        super(FFT_Loss, self).__init__()
        self.reduction = reduction

    def image_to_freq(self, image):
        """
        Convert image to frequency domain using 2D FFT.

        Args:
            image (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Frequency representation with real and imaginary parts stacked as extra channels.
        """
        freq = torch.fft.fft2(image)  # Compute 2D FFT
        freq = torch.stack([freq.real, freq.imag], dim=-1)  # Stack real and imaginary parts
        return freq

    def forward(self, targets, outputs):
        """
        Compute the FFT loss between target and output images.

        Args:
            targets (torch.Tensor): Ground truth image tensor of shape (B, C, H, W).
            outputs (torch.Tensor): Generated image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Computed FFT loss.
        """
        targets_freq = self.image_to_freq(targets)
        outputs_freq = self.image_to_freq(outputs)

        # Compute Mean Absolute Error (L1 loss) in frequency domain
        loss = F.l1_loss(targets_freq, outputs_freq, reduction=self.reduction)
        return loss
