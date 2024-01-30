import torch


def make_mask(lengths, size):
    mask = torch.arange(size).unsqueeze(0).repeat(len(lengths), 1)
    mask = mask < lengths.unsqueeze(1)
    return mask


def smape(pred, targ):
    return torch.mean(2 * torch.abs(pred - targ) / (torch.abs(pred) + torch.abs(targ)))


def normalize(seq_x, seq_y):
    with torch.no_grad():
        norm_consts = torch.max(seq_x[:, 0, :], dim=1).values.unsqueeze(1)
        seq_x[:, 0, :] /= norm_consts
        seq_y[:, :, 0] /= norm_consts
        return seq_x, seq_y, norm_consts


def denormalize(seq_x, seq_y, out, norm_consts):
    with torch.no_grad():
        seq_x[:, 0, :] *= norm_consts
        seq_y *= norm_consts
        out *= norm_consts
        return seq_x, seq_y, out
