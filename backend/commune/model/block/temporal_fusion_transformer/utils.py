import torch

def create_mask(size: int, lengths: torch.LongTensor, inverse: bool = False) -> torch.BoolTensor:
    """
    Create boolean masks of shape len(lenghts) x size.

    An entry at (i, j) is True if lengths[i] > j.

    Args:
        size (int): size of second dimension
        lengths (torch.LongTensor): tensor of lengths
        inverse (bool, optional): If true, boolean mask is inverted. Defaults to False.

    Returns:
        torch.BoolTensor: mask
    """

    if inverse:  # return where values are
        return torch.arange(size, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(-1)
    else:  # return where no values are
        return torch.arange(size, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(-1)

