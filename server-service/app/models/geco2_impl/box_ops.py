import torch
import torch.nn.functional as F


def boxes_with_scores(density_map, tlrb, sort=False, validate=False):
    B, C, _, _ = density_map.shape

    pooled = F.max_pool2d(density_map, 3, 1, 1)
    if validate:
        batch_thresh = (
            torch.max(density_map.reshape(B, -1), dim=-1).values.view(
                B, C, 1, 1
            )
            / 8
        )
    else:
        batch_thresh = torch.median(
            density_map.reshape(B, -1), dim=-1
        ).values.view(B, C, 1, 1)

    mask = (pooled == density_map) & (density_map > batch_thresh)

    out_batch = []
    ref_points_batch = []
    for i in range(B):
        bbox_scores = density_map[i, mask[i]]
        ref_points = mask[i].nonzero()[:, -2:]

        bbox_centers = ref_points / torch.tensor(
            mask.shape[2:], device=mask.device
        )

        tlrb_ = tlrb[i].permute(1, 2, 0)
        bbox_offsets = tlrb_[
            mask[i].permute(1, 2, 0).expand_as(tlrb_)
        ].reshape(-1, 4)

        sign = torch.tensor([-1, -1, 1, 1], device=mask.device)
        bbox_xyxy = bbox_centers.flip(-1).repeat(1, 2) + sign * bbox_offsets

        if sort:
            perm = torch.argsort(bbox_scores, descending=True)
            bbox_scores = bbox_scores[perm]
            bbox_xyxy = bbox_xyxy[perm]
            ref_points = ref_points[perm]

        out_batch.append(
            {
                "pred_boxes": bbox_xyxy.unsqueeze(0),
                "box_v": bbox_scores.unsqueeze(0),
            }
        )
        ref_points_batch.append(ref_points.T)

    return out_batch, ref_points_batch
