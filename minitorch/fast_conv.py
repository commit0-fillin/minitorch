from typing import Tuple
import numpy as np
from numba import njit, prange
from .autodiff import Context
from .tensor import Tensor
from .tensor_data import MAX_DIMS, Index, Shape, Strides, broadcast_index, index_to_position, to_index
from .tensor_functions import Function
to_index = njit(inline='always')(to_index)
index_to_position = njit(inline='always')(index_to_position)
broadcast_index = njit(inline='always')(broadcast_index)

def _tensor_conv1d(out: Tensor, out_shape: Shape, out_strides: Strides, out_size: int, input: Tensor, input_shape: Shape, input_strides: Strides, weight: Tensor, weight_shape: Shape, weight_strides: Strides, reverse: bool) -> None:
    """
    1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at left or right
    """
    batch, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels, in_channels, k_width = weight_shape

    for b in prange(batch):
        for oc in prange(out_channels):
            for ow in prange(out_width):
                out_pos = (
                    b * out_strides[0]
                    + oc * out_strides[1]
                    + ow * out_strides[2]
                )
                acc = 0.0
                for ic in prange(in_channels):
                    for kw in range(k_width):
                        w = weight_shape[2] - kw - 1 if reverse else kw
                        iw = ow + w - weight_shape[2] // 2
                        if 0 <= iw < width:
                            in_pos = (
                                b * input_strides[0]
                                + ic * input_strides[1]
                                + iw * input_strides[2]
                            )
                            w_pos = (
                                oc * weight_strides[0]
                                + ic * weight_strides[1]
                                + kw * weight_strides[2]
                            )
                            acc += input[in_pos] * weight[w_pos]
                out[out_pos] = acc
tensor_conv1d = njit(parallel=True)(_tensor_conv1d)

class Conv1dFun(Function):

    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """
        Compute a 1D Convolution

        Args:
            ctx : Context
            input : batch x in_channel x w
            weight : out_channel x in_channel x kw

        Returns:
            batch x out_channel x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        out_w = w - kw + 1
        
        out = input.zeros((batch, out_channels, out_w))
        tensor_conv1d(
            out._tensor,
            input._tensor,
            weight._tensor,
            False
        )
        return out
conv1d = Conv1dFun.apply

def _tensor_conv2d(out: Tensor, out_shape: Shape, out_strides: Strides, out_size: int, input: Tensor, input_shape: Shape, input_strides: Strides, weight: Tensor, weight_shape: Shape, weight_strides: Strides, reverse: bool) -> None:
    """
    2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right
    """
    batch, out_channels, out_height, out_width = out_shape
    batch, in_channels, in_height, in_width = input_shape
    out_channels, in_channels, k_height, k_width = weight_shape

    for b in prange(batch):
        for oc in prange(out_channels):
            for oh in prange(out_height):
                for ow in prange(out_width):
                    out_pos = (
                        b * out_strides[0]
                        + oc * out_strides[1]
                        + oh * out_strides[2]
                        + ow * out_strides[3]
                    )
                    acc = 0.0
                    for ic in prange(in_channels):
                        for kh in range(k_height):
                            for kw in range(k_width):
                                h = weight_shape[2] - kh - 1 if reverse else kh
                                w = weight_shape[3] - kw - 1 if reverse else kw
                                ih = oh + h - weight_shape[2] // 2
                                iw = ow + w - weight_shape[3] // 2
                                if 0 <= ih < in_height and 0 <= iw < in_width:
                                    in_pos = (
                                        b * input_strides[0]
                                        + ic * input_strides[1]
                                        + ih * input_strides[2]
                                        + iw * input_strides[3]
                                    )
                                    w_pos = (
                                        oc * weight_strides[0]
                                        + ic * weight_strides[1]
                                        + kh * weight_strides[2]
                                        + kw * weight_strides[3]
                                    )
                                    acc += input[in_pos] * weight[w_pos]
                    out[out_pos] = acc
tensor_conv2d = njit(parallel=True, fastmath=True)(_tensor_conv2d)

class Conv2dFun(Function):

    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """
        Compute a 2D Convolution

        Args:
            ctx : Context
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw

        Returns:
            (:class:`Tensor`) : batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape
        out_h = h - kh + 1
        out_w = w - kw + 1
        
        out = input.zeros((batch, out_channels, out_h, out_w))
        tensor_conv2d(
            out._tensor,
            input._tensor,
            weight._tensor,
            False
        )
        return out
conv2d = Conv2dFun.apply
