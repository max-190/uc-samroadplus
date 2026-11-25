
import numpy as np
import torch

from model import extract_point


def _extract_point_uniform(x1,y1,x2,y2,image,num_points):
    H, W = image.shape[-2:]
    x_values = torch.linspace(0, 1, steps=num_points).unsqueeze(0).unsqueeze(0).to(image.device)
    y_values = torch.linspace(0, 1, steps=num_points).unsqueeze(0).unsqueeze(0).to(image.device)#uniform sampling

    x_interp = x1.unsqueeze(-1) + (x2 - x1).unsqueeze(-1) * x_values
    y_interp = y1.unsqueeze(-1) + (y2 - y1).unsqueeze(-1) * y_values

    x_interp = torch.clamp(x_interp.long(), min=0, max=W-1)
    y_interp = torch.clamp(y_interp.long(), min=0, max=H-1)

    x_plus_1 = torch.clamp(x_interp + 1, max=W-1)
    y_plus_1 = torch.clamp(y_interp + 1, max=H-1)

    x_final = torch.cat([x_interp,x_interp,x_plus_1], dim=-1)
    y_final = torch.cat([y_interp,y_plus_1,y_interp], dim=-1)

    return(x_final,y_final)# sample points


def uniform(points1, points2, image, sample_n_inside=20, sample_n_outside=15):
    """Using linear interpolation to get pixels between two points in batch."""
    B, N, _ = points1.shape  # B: batch size, N: number of point pairs
    H, W = image.shape[-2:]
    height, width = H,W
    extend_length=8 #extend length
    batch_A = points1
    batch_B = points2
    # direction vector
    directions = batch_B - batch_A  # (N, 2)

    lengths = torch.norm(directions, dim=2, keepdim=True)  # (N, 1)
    lengths = lengths.masked_fill(lengths == 0, 1e-8)
    directions_norm = directions / lengths  # (N, 2)

    extended_A = batch_A - directions_norm * extend_length  # (N, 2)
    extended_B = batch_B + directions_norm * extend_length  # (N, 2)
      # Round the coordinates to integers
    extended_A = torch.round(extended_A).long()
    extended_B = torch.round(extended_B).long()

    extended_A[:, 0] = extended_A[:, 0].clamp(0, width - 1)
    extended_A[:, 1] = extended_A[:, 1].clamp(0, height - 1)
    extended_B[:, 0] = extended_B[:, 0].clamp(0, width - 1)
    extended_B[:, 1] = extended_B[:, 1].clamp(0, height - 1)

    extend_x1,extend_y1 = extended_A[...,0],extended_A[...,1]
    extend_x2,extend_y2 = extended_B[...,0],extended_B[...,1]

    x1, y1 = points1[..., 0], points1[..., 1]
    x2, y2 = points2[..., 0], points2[..., 1]

    x_final_1,y_final_1  = _extract_point_uniform(extend_x1,extend_y1,x1,y1,image,num_points=sample_n_outside)#sample extend point one side
    x_final,y_final  = _extract_point_uniform(x1,y1,x2,y2,image,num_points=sample_n_inside) #sample between point
    x_final_2,y_final_2  = _extract_point_uniform(extend_x2,extend_y2,x2,y2,image,num_points=sample_n_outside)#sample extend point other side

    features1 = image[np.arange(B)[:, None, None], x_final_1,  y_final_1]
    features = image[np.arange(B)[:, None, None], x_final,  y_final]
    features2 = image[np.arange(B)[:, None, None], x_final_2,  y_final_2]#extract mask feature
    features = torch.concat([features1,features,features2], dim=2)

    return features


def normal(points1, points2, image, sample_n_inside=20, sample_n_outside=15):
    """
        Sample points between and on either side of the points according to a normal distribution.
        (Sample more frequently towards the center between the points)
    """
    raise NotImplementedError()


def bimodal(points1, points2, image, sample_n_inside=20, sample_n_outside=15):
    """
        Sample points between and on either side of the points according to a bimodal distribution.
        (Sample more frequently around the points)
    """
    raise NotImplementedError()


def edge_detection(points1, points2, image, sample_n_inside=20, sample_n_outside=15):
    """
        Sample points between and on either side of the points, sampling more densely around occlusion 'edges'.
    """

    raise NotImplementedError()


# TODO: More strategies?