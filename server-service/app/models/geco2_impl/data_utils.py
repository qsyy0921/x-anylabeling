import torch


def resize_and_pad(img, bboxes, size=1024.0):
    """Resize image and exemplar boxes to fit within a square canvas.

    Args:
        img: Tensor of shape (C, H, W) in float.
        bboxes: Tensor of shape (N, 4) with exemplar box coordinates in
            the original image frame.
        size: Target canvas size in pixels (default 1024).

    Returns:
        Tuple of (padded_img, resized_bboxes, scaling_factor).
    """
    channels, original_height, original_width = img.shape
    longer_dimension = max(original_height, original_width)
    scaling_factor = size / longer_dimension
    scaled_bboxes = bboxes * scaling_factor
    a_dim = (
        (scaled_bboxes[:, 2] - scaled_bboxes[:, 0]).mean()
        + (scaled_bboxes[:, 3] - scaled_bboxes[:, 1]).mean()
    ) / 2
    scaling_factor = min(1.0, 80 / a_dim.item()) * scaling_factor
    resized_img = torch.nn.functional.interpolate(
        img.unsqueeze(0),
        scale_factor=scaling_factor,
        mode="bilinear",
        align_corners=False,
    )

    size = int(size)
    pad_height = max(0, size - resized_img.shape[2])
    pad_width = max(0, size - resized_img.shape[3])
    padded_img = torch.nn.functional.pad(
        resized_img, (0, pad_width, 0, pad_height), mode="constant", value=0
    )[0]

    bboxes = bboxes * torch.tensor(
        [scaling_factor, scaling_factor, scaling_factor, scaling_factor],
        device=bboxes.device,
    )
    return padded_img, bboxes, scaling_factor
