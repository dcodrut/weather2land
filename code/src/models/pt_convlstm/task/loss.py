import torch
import torch.nn.functional as F
from torch import nn


class MaskedDist(nn.Module):
    def __init__(self, distance_type="L2"):
        super(MaskedDist, self).__init__()
        self.distance_type = distance_type

    def forward(self, preds, targets, mask):
        assert (preds.shape == targets.shape)
        predsmasked = preds * mask
        targetsmasked = targets * mask

        if self.distance_type == "L2":
            return F.mse_loss(predsmasked, targetsmasked, reduction="sum") / ((mask > 0).sum() + 1)
        elif self.distance_type == "L1":
            return F.l1_loss(predsmasked, targetsmasked, reduction="sum") / ((mask > 0).sum() + 1)


class TempMaskedDist(nn.Module):
    def __init__(self, distance_type="L2", kernel_size=5):
        super().__init__()
        self.distance = MaskedDist(distance_type=distance_type)
        self.kernel_size = kernel_size
        self.avg_filter = torch.ones((1, 1, self.kernel_size, self.kernel_size)).cuda() / (self.kernel_size ** 2)

    def forward(self, preds, targets, mask):
        assert (preds.shape == targets.shape)

        temp_diff_preds = preds[:, 1:, ...] - preds[:, :-1, ...]

        # smooth the targets
        temp_diff_targets = targets[:, 1:, ...] - targets[:, :-1, ...]
        b, t, c, h, w = temp_diff_targets.shape
        temp_diff_targets = temp_diff_targets.view(b, c * t, h, w)
        avg_filter = self.avg_filter.repeat((c * t, 1, 1, 1))
        temp_diff_targets = F.conv2d(temp_diff_targets, weight=avg_filter, stride=1,
                                     padding=(self.kernel_size - 1) // 2, groups=c * t)
        temp_diff_targets = temp_diff_targets.view((b, t, c, h, w))

        mask_time = mask[:, 1:, ...] * mask[:, :-1, ...]

        return self.distance(temp_diff_preds, temp_diff_targets, mask_time)


LOSSES = {
    "masked": MaskedDist,
}


class BaseLoss(nn.Module):
    def __init__(self, setting: dict):
        super().__init__()

        self.distance = LOSSES[setting["name"]](**setting["args"])

    def forward(self, preds, batch, aux, current_step=None):
        logs = {}

        targs = batch["dynamic"][0][:, -preds.shape[1]:, ...]
        masks = batch["dynamic_mask"][0][:, -preds.shape[1]:, ...]

        dist = self.distance(preds, targs, masks)

        logs["distance"] = dist

        loss = dist

        logs["loss"] = loss

        return loss, logs


class MaskedDistAndTempDistLoss(nn.Module):
    def __init__(self, setting: dict):
        super().__init__()

        self.distance = MaskedDist(distance_type=setting["args"]["distance_type"])
        self.temp_dist = TempMaskedDist(distance_type=setting["args"]["distance_type"])
        self.alpha = setting["args"]["alpha"]

    def forward(self, preds, batch, aux, current_step=None):
        logs = {}

        targs = batch["dynamic"][0][:, -preds.shape[1]:, ...]
        masks = batch["dynamic_mask"][0][:, -preds.shape[1]:, ...]

        dist = self.distance(preds, targs, masks)
        logs["distance"] = dist

        temp_dist = self.temp_dist(preds, targs, masks)
        logs["temp_dist"] = temp_dist

        loss = dist + self.alpha * temp_dist
        logs["loss"] = loss

        return loss, logs


def setup_loss(args):
    if args["name"] == "masked":
        return BaseLoss(args)
    elif args["name"] == "masked_dist_and_temp_dist":
        return MaskedDistAndTempDistLoss(args)
    else:
        raise NotImplemented
