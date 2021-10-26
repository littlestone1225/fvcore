# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
from torch.nn import functional as F
import numpy as np

'''
def split_targets(
    targets,
    idx
):
    batch_size, class_num = targets.size()
    split_target = torch.zeros(targets.size())
    if targets.get_device() >= 0:
        split_target = split_target.to("cuda:{}".format(targets.get_device()))
    nonzero_idx = torch.nonzero(targets[:,idx]>0)
    row = nonzero_idx[:,0]
    col = idx
    split_target[row, col] = 1
    return split_target
'''
print_log = False

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    if print_log:
        num = 5
        idx = torch.zeros(2, dtype=torch.int)
        nonzero_idx = torch.nonzero(targets)
        print(" = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ")
        print("fg = {} ; bg = {} ; total = {}".format(nonzero_idx.size(0), targets.size(0)-nonzero_idx.size(0), targets.size(0)))

    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)

    loss = ce_loss * ((1 - p_t) ** gamma)

    '''
    target_bridge = split_targets(targets, torch.tensor([0]))
    target_empty = split_targets(targets, torch.tensor([1]))
    target_other = targets - target_bridge - target_empty
    '''
    if alpha >= 0:
        # alpha_t = alpha * targets + (1 - alpha) * (1 - targets) # v6_17, v6_18, v11_3
        # alpha_t = 0.99 * targets + 0.3 * (1 - targets) # v6_4, v6_4_1, v6_4_3, v6_5, v6_10, v6_15, v6_16, v6_19, v11_1, v11_2, v11_4
        # alpha_t = 0.99 * targets + 0.6 * (1 - targets) # v6_4_2
        # alpha_t = 1.5 * targets + 0.3 * (1 - targets) # v6_4_4
        alpha_t = 1.5 * targets + 0.6 * (1 - targets) # v6_4_6, v6_4_7, v6_4_10, v6_4_11, v6_4_12, v6_4_14
        # alpha_t = 1.5 * targets + 0.3 * (1 - targets) # v6_4_13
        # alpha_t = 0.4 * targets + 0.6 * (1 - targets) # v6_4_5
        # alpha_t = 1.5 * targets + 0.3 * (1 - targets) # v6_13
        # alpha_t = 0.99 * target_bridge + 1.1 * target_empty + 0.6 * target_other + 0.3 * (1 - targets) # v6_7
        # alpha_t = 1.3 * target_bridge + 1.8 * target_empty + 0.6 * target_other + 0.4 * (1 - targets) # v6_8
        # alpha_t = 1.5 * target_bridge + 2.0 * target_empty + 0.6 * target_other + 0.5 * (1 - targets) # v6_14
        loss = alpha_t * loss

    if print_log and nonzero_idx.size(0) > 0:
        if nonzero_idx.size(0) < num:
            num = nonzero_idx.size(0)
        idx = nonzero_idx[0]
        print("inputs[idx[0]] = \n{}".format(inputs[idx[0]:idx[0]+num]))
        print("targets[idx[0]] = \n{}".format(targets[idx[0]:idx[0]+num]))
        print("p[idx[0]] = \n{}".format(p[idx[0]:idx[0]+num]))
        print("ce_loss[idx[0]] = \n{}".format(ce_loss[idx[0]:idx[0]+num]))
        print("p_t[idx[0]] = \n{}".format(p_t[idx[0]:idx[0]+num]))
        print("loss[idx[0]] = \n{}".format(loss[idx[0]:idx[0]+num]))

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    if print_log and nonzero_idx.size(0) > 0:
        print("loss = {}".format(loss))

    if print_log:
        print(" ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ \n")
    return loss


sigmoid_focal_loss_jit = torch.jit.script(
    sigmoid_focal_loss
)  # type: torch.jit.ScriptModule


def sigmoid_focal_loss_star(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 1,
    reduction: str = "none",
) -> torch.Tensor:
    """
    FL* described in RetinaNet paper Appendix: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Gamma parameter described in FL*. Default = 1 (no weighting).
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    shifted_inputs = gamma * (inputs * (2 * targets - 1))
    loss = -(F.logsigmoid(shifted_inputs)) / gamma

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss *= alpha_t

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


sigmoid_focal_loss_star_jit = torch.jit.script(
    sigmoid_focal_loss_star
)  # type: torch.jit.ScriptModule
