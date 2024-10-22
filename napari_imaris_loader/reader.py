# napari_imaris_loader.py

import os
import numpy as np
import dask.array as da
import napari
from magicgui import magic_factory
from napari_plugin_engine import napari_hook_implementation
from napari.layers import Image
from typing import List, Tuple, Any

# 确保已安装 imaris_ims_file_reader 库
# pip install imaris_ims_file_reader
from imaris_ims_file_reader.ims import ims

def ims_reader(path: str, resLevel: int = 0, colorsIndependant: bool = False, preCache: bool = False) -> List[Tuple[Any, dict]]:
    """
    读取 .ims 文件并返回可用于 napari 的数据和元数据。
    """
    # 确保启用了异步加载
    os.environ["NAPARI_ASYNC"] = "1"

    # 加载 ims 文件
    squeeze_output = False  # 不进行 squeeze，以保持维度
    imsClass = ims(path, squeeze_output=squeeze_output)

    # 输出可用的分辨率级别数量
    print(f"Available resolution levels: {imsClass.ResolutionLevels}")

    # 尝试从最低分辨率级别提取对比度限制
    try:
        # 提取最低分辨率级别并计算对比度限制
        minResolutionLevel = imsClass[imsClass.ResolutionLevels - 1, :, :, :, :, :]
        minContrast = minResolutionLevel[minResolutionLevel > 0].min()
        maxContrast = minResolutionLevel.max()
        contrastLimits = [minContrast, maxContrast]
    except Exception:
        # 根据数据类型设置默认对比度限制
        if imsClass.dtype == np.dtype('uint16'):
            contrastLimits = [0, 65535]
        elif imsClass.dtype == np.dtype('uint8'):
            contrastLimits = [0, 255]
        else:
            contrastLimits = [0, 1]

    # 准备通道名称
    channelNames = [f'Channel {cc}' for cc in range(imsClass.Channels)]

    # 加载每个分辨率级别的数据
    data = []
    for rr in range(imsClass.ResolutionLevels):
        print(f'Loading resolution level {rr}')
        data.append(ims(path, ResolutionLevelLock=rr, cache_location=imsClass.cache_location, squeeze_output=squeeze_output))

    # 将数据转换为具有适当分块的 Dask 数组
    for idx in range(len(data)):
        data[idx] = da.from_array(
            data[idx],
            chunks=data[idx].chunks,
            fancy=False
        )

    # 可选地限制分辨率级别的数量
    if isinstance(resLevel, int):
        if resLevel < 0 or resLevel >= len(data):
            raise ValueError(f'所选分辨率级别无效：请选择 0 到 {len(data) - 1} 之间的值')
        data = data[resLevel:]

    # 在裁剪数据之后设置 multiscale 参数
    meta = {
        "contrast_limits": contrastLimits,
        "name": channelNames,
        "metadata": {
            'fileName': imsClass.filePathComplete,
            'resolutionLevels': imsClass.ResolutionLevels
        }
    }
    meta["multiscale"] = True if isinstance(data, list) else False

    # 打印数据形状
    for idx, level_data in enumerate(data):
        print(f"Resolution level {idx + resLevel} data shape: {level_data.shape}")

    # 确定通道轴
    channel_axis = None
    for idx, dim in enumerate(data[0].shape):
        if dim == imsClass.Channels and imsClass.Channels > 1:
            channel_axis = idx
            break

    meta['channel_axis'] = channel_axis

    # 确保数据形状一致，并包含通道轴
    for idx in range(len(data)):
        if channel_axis is not None and data[idx].ndim <= channel_axis:
            data[idx] = np.expand_dims(data[idx], axis=channel_axis)

    # 处理独立的颜色（通道）
    if colorsIndependant and channel_axis is not None:
        channelData = []
        num_channels = data[0].shape[channel_axis]
        for cc in range(num_channels):
            singleChannel = []
            for dd in data:
                indexer = [slice(None)] * dd.ndim
                indexer[channel_axis] = cc
                singleChannel.append(dd[tuple(indexer)])
            channelData.append(singleChannel)

        del meta['channel_axis']  # 从元数据中删除 channel_axis

        # 为每个通道创建元数据
        metaData = []
        for cc in range(num_channels):
            # 根据每个通道调整 scale
            data_shape = channelData[cc][0].shape
            extra_dims = len(data_shape) - len(imsClass.resolution)
            singleChannelScale = (1,) * extra_dims + tuple(imsClass.resolution)

            # 判断当前通道的数据是否为多尺度
            is_multiscale = True if isinstance(channelData[cc], list) else False

            singleChannelMeta = {
                'contrast_limits': meta['contrast_limits'],
                'multiscale': is_multiscale,
                'metadata': meta['metadata'],
                'scale': singleChannelScale,
                'name': meta['name'][cc]
            }
            metaData.append(singleChannelMeta)

        # 准备最终输出
        finalOutput = []
        for dd, mm in zip(channelData, metaData):
            finalOutput.append((dd, mm))
        return finalOutput
    else:
        # 调整组合数据的 scale
        data_shape = data[0].shape
        extra_dims = len(data_shape) - len(imsClass.resolution)
        scale = (1,) * extra_dims + tuple(imsClass.resolution)

        # 如果指定了 channel_axis，调整每个通道的数据的 scale
        if channel_axis is not None:
            # 拆分后，每个通道的数据将减少一个维度
            per_channel_shape = data_shape[:channel_axis] + data_shape[channel_axis + 1:]
            per_channel_extra_dims = len(per_channel_shape) - len(imsClass.resolution)
            per_channel_scale = (1,) * per_channel_extra_dims + tuple(imsClass.resolution)
            meta["scale"] = per_channel_scale
        else:
            meta["scale"] = scale

        meta["multiscale"] = True if isinstance(data, list) else False

        return [(data, meta)] if meta["multiscale"] else [(data[0], meta)]

@napari_hook_implementation
def napari_get_reader(path):
    if isinstance(path, str) and os.path.splitext(path)[1].lower() == '.ims':
        return ims_reader