from typing import Tuple, List, Union, Optional, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

from scirex.torch.sciml.builders.spectral_convolution import SpectralConv
from scirex.torch.sciml.builders.embeddings import GridEmbeddingND, GridEmbedding2D
from scirex.torch.sciml.builders.channel_mlp import ChannelMLP
from scirex.torch.sciml.builders.skip_connections import skip_connection
from scirex.torch.sciml.builders.normalization import AdaIN, InstanceNorm, BatchNorm
from scirex.torch.sciml.builders.padding import DomainPadding
from scirex.torch.sciml.builders.complex import ComplexValued
from scirex.torch.sciml.models.base_model import BaseModel

Number = Union[float, int]

class FNOBlock(nn.Module):
    """
    A single Fourier Neural Operator Block.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int, ...],
        output_scaling_factor: Union[Number, List[Number]] = None,
        use_channel_mlp: bool = False,
        channel_mlp_expansion: float = 0.5,
        channel_mlp_dropout: float = 0.0,
        non_linearity: nn.Module = F.gelu,
        stabilizer: str = None,
        norm: Literal["ada_in", "group_norm", "instance_norm", "batch_norm"] = None,
        fno_skip: str = "linear",
        channel_mlp_skip: str = "soft-gating",
        preactivation: bool = False,
        fno_block_precision: str = "full",
        rank: float = 1.0,
        fixed_rank_modes: bool = False,
        implementation: str = "factorized",
        separable: bool = False,
        factorization: str = None,
        decomposition_kwargs: dict = None,
        complex_data: bool = False
    ):
        super().__init__()
        self.n_dim = len(n_modes)
        self.output_scaling_factor = output_scaling_factor
        self.preactivation = preactivation
        self.complex_data = complex_data
        
        self.spectral_conv = SpectralConv(
            in_channels=in_channels,
            out_channels=out_channels,
            n_modes=n_modes,
            resolution_scaling_factor=output_scaling_factor,
            fno_block_precision=fno_block_precision,
            rank=rank,
            factorization=factorization,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            separable=separable,
            decomposition_kwargs=decomposition_kwargs,
            bias=False
        )
        
        self.skip = skip_connection(
            in_features=in_channels,
            out_features=out_channels,
            skip_type=fno_skip,
            n_dim=self.n_dim,
            bias=True
        )

        self.norm = self._get_norm(norm, in_channels if preactivation else out_channels)
        self.non_linearity = non_linearity
        
        self.stabilizer = None
        if stabilizer == "tanh":
            self.stabilizer = torch.tanh

        self.channel_mlp = None
        if use_channel_mlp:
            self.channel_mlp = ChannelMLP(
                in_channels=out_channels,
                out_channels=out_channels,
                hidden_channels=int(out_channels * channel_mlp_expansion),
                n_layers=2,
                n_dim=self.n_dim,
                non_linearity=non_linearity,
                dropout=channel_mlp_dropout
            )
            self.channel_mlp_skip = skip_connection(
                in_features=out_channels,
                out_features=out_channels,
                skip_type=channel_mlp_skip,
                n_dim=self.n_dim,
                bias=True
            )

    def _get_norm(self, norm_type, channels):
        if norm_type is None:
            return None
        if norm_type == "instance_norm":
            return InstanceNorm(affine=False)
        elif norm_type == "batch_norm":
            return BatchNorm(n_dim=self.n_dim, num_features=channels)
        elif norm_type == "group_norm":
            return nn.GroupNorm(num_groups=1, num_channels=channels)
        return None

    def forward(self, x, output_shape=None, embedding=None):
        if self.preactivation and self.norm is not None:
             x = self.norm(x)

        x_fnoir = self.spectral_conv(x, output_shape=output_shape)
        
        if output_shape is not None and list(x.shape[2:]) != list(output_shape):
             x_res = F.interpolate(x, size=output_shape, mode='bilinear' if self.n_dim==2 else 'trilinear', align_corners=False)
        else:
             x_res = x

        x_skip = self.skip(x_res)
        x = x_fnoir + x_skip

        if not self.preactivation and self.norm is not None:
             if isinstance(self.norm, AdaIN) and embedding is not None:
                  self.norm.set_embedding(embedding)
             x = self.norm(x)

        if self.stabilizer is not None:
            x = self.stabilizer(x)

        if self.non_linearity is not None:
            x = self.non_linearity(x)

        if self.channel_mlp is not None:
            x_mlp = self.channel_mlp(x)
            if self.channel_mlp_skip is None:
                x = x_mlp
            else:
                x = x + self.channel_mlp_skip(x_mlp)
                
        return x


class FNO(BaseModel):
    """
    Unified Fourier Neural Operator (FNO) model.
    """
    def __init__(
        self,
        n_modes: Tuple[int, ...],
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        n_layers: int = 4,
        lifting_channel_ratio: Number = 2,
        projection_channel_ratio: Number = 2,
        positional_embedding: Union[str, nn.Module] = "grid",
        non_linearity: nn.Module = F.gelu,
        norm: Literal["ada_in", "group_norm", "instance_norm", "batch_norm"] = None,
        complex_data: bool = False,
        use_channel_mlp: bool = True,
        channel_mlp_dropout: float = 0,
        channel_mlp_expansion: float = 0.5,
        channel_mlp_skip: Literal["linear", "identity", "soft-gating", None] = "soft-gating",
        fno_skip: Literal["linear", "identity", "soft-gating", None] = "linear",
        resolution_scaling_factor: Union[Number, List[Number]] = None,
        domain_padding: Union[Number, List[Number]] = None,
        fno_block_precision: str = "full",
        stabilizer: str = None,
        max_n_modes: Tuple[int, ...] = None,
        factorization: str = None,
        rank: float = 1.0,
        fixed_rank_modes: bool = False,
        implementation: str = "factorized",
        decomposition_kwargs: dict = None,
        separable: bool = False,
        preactivation: bool = False,
        conv_module: nn.Module = SpectralConv,
    ):
        super().__init__()
        
        self.n_modes = n_modes
        self.n_dim = len(n_modes)
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.domain_padding_val = domain_padding
        self.output_scaling_factor = resolution_scaling_factor
        self.complex_data = complex_data

        if positional_embedding == "grid":
            self.pos_embed = GridEmbeddingND(in_channels, self.n_dim)
            lifting_in = self.pos_embed.out_channels
        elif isinstance(positional_embedding, nn.Module):
             self.pos_embed = positional_embedding
             lifting_in = self.pos_embed.out_channels
        else:
             self.pos_embed = None
             lifting_in = in_channels

        lifting_channels = int(hidden_channels * lifting_channel_ratio)
        self.lifting = ChannelMLP(
            in_channels=lifting_in,
            out_channels=hidden_channels,
            hidden_channels=lifting_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity
        )
        if complex_data:
            self.lifting = ComplexValued(self.lifting)

        self.domain_padding = None
        if domain_padding is not None:
            self.domain_padding = DomainPadding(domain_padding=domain_padding, resolution_scaling_factor=resolution_scaling_factor)

        blocks = []
        for i in range(n_layers):
            if resolution_scaling_factor is not None:
                if isinstance(resolution_scaling_factor, list):
                    scaling = resolution_scaling_factor[i]
                else: 
                     scaling = resolution_scaling_factor
            else:
                 scaling = None

            blocks.append(FNOBlock(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                n_modes=n_modes,
                output_scaling_factor=scaling,
                use_channel_mlp=use_channel_mlp,
                channel_mlp_expansion=channel_mlp_expansion,
                channel_mlp_dropout=channel_mlp_dropout,
                non_linearity=non_linearity,
                stabilizer=stabilizer,
                norm=norm,
                fno_skip=fno_skip,
                channel_mlp_skip=channel_mlp_skip,
                preactivation=preactivation,
                fno_block_precision=fno_block_precision,
                rank=rank,
                fixed_rank_modes=fixed_rank_modes,
                implementation=implementation,
                separable=separable,
                factorization=factorization,
                decomposition_kwargs=decomposition_kwargs,
                complex_data=complex_data
            ))
        self.fno_blocks = nn.ModuleList(blocks)

        projection_channels = int(hidden_channels * projection_channel_ratio)
        self.projection = ChannelMLP(
            in_channels=hidden_channels,
            out_channels=out_channels,
            hidden_channels=projection_channels,
            n_layers=2,
            n_dim=self.n_dim,
            non_linearity=non_linearity
        )
        if complex_data:
            self.projection = ComplexValued(self.projection)

    def forward(self, x, output_shape=None, embedding=None, **kwargs):
        if self.pos_embed is not None:
             x = self.pos_embed(x)

        x = self.lifting(x)
        
        if self.domain_padding is not None:
             x = self.domain_padding.pad(x)
        
        if output_shape is None:
             out_shapes = [None]*self.n_layers
        elif isinstance(output_shape, list):
             out_shapes = output_shape
        else:
             out_shapes = [None]*(self.n_layers-1) + [output_shape]

        for i, block in enumerate(self.fno_blocks):
             x = block(x, output_shape=out_shapes[i], embedding=embedding)

        if self.domain_padding is not None:
             x = self.domain_padding.unpad(x)
        
        x = self.projection(x)
        
        return x
