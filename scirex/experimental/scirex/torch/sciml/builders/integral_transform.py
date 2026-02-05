import torch
from torch import nn
import torch.nn.functional as F

from .channel_mlp import LinearChannelMLP

class IntegralTransform(nn.Module):

    """Integral Kernel Transform (GNO).
    
    It computes one of the following:
        (a) \\int_{A(x)} k(x, y) dy
        (b) \\int_{A(x)} k(x, y) * f(y) dy
        (c) \\int_{A(x)} k(x, y, f(y)) dy
        (d) \\int_{A(x)} k(x, y, f(y)) * f(y) dy

    x : Points for which the output is defined

    y : Points for which the input is defined
    A(x) : A subset of all points y (depending on\
        each x) over which to integrate

    k : A kernel parametrized as a MLP (LinearChannelMLP)
    
    f : Input function to integrate against given\
        on the points y

    If f is not given, a transform of type (a)
    is computed. Otherwise transforms (b), (c),
    or (d) are computed. The sets A(x) are specified
    as a graph in CRS format.

    Purpose: Transforms input function f(y) → output function at x via neighbor-weighted kernel integration.

    INPUT FORMAT:
    y:              [n_points, coord_dim]           e.g., [1000, 2] for 2D points
    neighbors:      dict                           e.g., {"neighbors_index": [8000], "neighbors_row_splits": [1001]}
    x:              [m_points, coord_dim]          e.g., [1000, 2] (usually x=y)
    f_y:            [batch, n_points, in_channels] or [n_points, in_channels]  e.g., [4, 1000, 1]

    Neighbors dict breakdown:
    neighbors["neighbors_index"]:  [total_neighbors]     e.g., [8000] (all neighbor indices)
    neighbors["neighbors_row_splits"]: [m_points+1]     e.g., [0,5,12,...,8000] (CSR format)

    OUTPUT FORMAT:
    Shape: [batch, m_points, out_channels] or [m_points, out_channels]
    Example: [4, 1000, 12]  # Batch=4, 1000 query points, 12 output channels

    Data content: Output function values at points x
    - out[i,b] = sum_{j∈neighbors(i)} k(x_i, y_j) * f(y_j) * weights_j
    - Each channel = one output feature dimension at query point x_i


    INTERNAL DATA FLOW:
    1. rep_features = y[neighbors["neighbors_index"]]        [8000, 2]   # Neighbor coordinates
    2. self_features = repeat_interleave(x, num_reps)        [8000, 2]   # Query coords per neighbor
    3. agg_features = cat([rep_features, self_features])     [8000, 128] # [y_pos(64)+x_pos(64)]
    4. If nonlinear: cat([agg_features, f_y_neighbors])      [8000, 129] # + f_y(1)
    5. kernel_weights = channel_mlp(agg_features)            [8000, 12]
    6. If linear: kernel_weights *= f_y_neighbors            [8000, 12]
    7. segment_csr(kernel_weights, splits) →                [4, 1000, 12]

    IntegralTransform approximates continuous kernel integrals over discrete neighbor graphs using Monte-Carlo summation. 
    It computes output(x_i) = sum_{y_j in neighbors(x_i)} k(x_i, y_j) * f(y_j) where k is parameterized by LinearChannelMLP.

    What does it do:
    For each query point x_i, finds its neighbors y_j, computes kernel weights k(x_i, y_j) via MLP, 
    multiplies by input values f(y_j), then sums across all neighbors to get output at x_i.
    """

    def __init__(
        self,
        channel_mlp=None,
        channel_mlp_layers=None,
        channel_mlp_non_linearity=F.gelu,
        transform_type='linear',
        weighting_fun=None,
        reduction="sum",
        use_torch_scatter=False,
    ):
        super.__init__()

        assert channel_mlp is not None or channel_mlp_layers is not None

        self.reduction = reduction # how to combine neighbor "sum" or "mean"
        self.transform_type = transform_type # "linear" or "nonlinear"
        self.weighting_fun = weighting_fun # function to transform neighbor weights
        self.use_torch_scatter = use_torch_scatter

        if (
            self.transform_type != "linear_kernel_only",
            self.transform_type != "linear",
            self.transform_type != "nonlinear",
            self.transform_type != "nonlinear_kernel_only",
        ):
            raise ValueError(
                f"Got transform type = {transform_type}, must be one of 'linear', 'nonlinear', 'linear_kernel_only', 'nonlinear_kernel_only'"
            )
        
        if channel_mlp is None:
            self.channel_mlp = LinearChannelMLP(
                layers=channel_mlp_layers, non_linearity=channel_mlp_non_linearity
            )

        else:
            self.channel_mlp = channel_mlp

        self.weighting_fn = weighting_fun

    def forward(self, y, neighbors, x=None, f_y=None, weights=None):
        pass
