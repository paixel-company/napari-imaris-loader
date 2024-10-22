import napari
from magicgui import magic_factory
from napari_plugin_engine import napari_hook_implementation
from .reader import ims_reader
from typing import List
from napari.layers import Image
from napari.types import LayerDataTuple

@magic_factory(
    auto_call=False,
    call_button="Update"
)
def resolution_change(
    viewer: napari.Viewer,
    lowest_resolution_level: int = 0
):
    """
    该面板提供了一个工具，用于在选择最低分辨率级别后重新加载 IMS 数据，
    该级别将包含在多分辨率系列中。较高的数字（即金字塔顶部）=较低的分辨率。

    这对于 3D 渲染很重要。如果您希望更高分辨率的 3D 渲染，可以选择较低的数字，更新视图，然后选择 3D 渲染。
    """

    # 查找元数据中包含 'fileName' 的第一个 IMS 图层
    ims_layer = None
    for layer in viewer.layers:
        if isinstance(layer, Image) and 'fileName' in layer.metadata:
            ims_layer = layer
            break

    if ims_layer is None:
        print("未找到元数据中包含 'fileName' 的 IMS 图层。")
        return

    # 获取可用的分辨率级别数量
    available_levels = ims_layer.metadata.get('resolutionLevels', None)
    if available_levels is None:
        print("无法获取可用的分辨率级别数量。")
        return

    # 验证 lowest_resolution_level 是否在有效范围内
    if lowest_resolution_level < 0 or lowest_resolution_level >= available_levels:
        print(f"所选的分辨率级别无效。请选择 0 到 {available_levels - 1} 之间的值。")
        return

    # 使用加载函数加载 IMS 文件的数据
    try:
        tupleOut = ims_reader(
            ims_layer.metadata['fileName'],
            colorsIndependant=True,
            resLevel=lowest_resolution_level
        )
    except ValueError as e:
        print(e)
        return

    # 确定从 IMS 文件中提取的通道名称
    channelNames = [tt[1]['name'] for tt in tupleOut]

    # 为避免插值和轴不匹配错误，将 viewer 强制为 2D 模式
    if viewer.dims.ndisplay == 3:
        viewer.dims.ndisplay = 2

    # 更新元数据并删除旧图层
    for num, channel_name in enumerate(channelNames):
        if channel_name in viewer.layers:
            layer = viewer.layers[channel_name]
            tmp = {
                'opacity': layer.opacity,
                'gamma': layer.gamma,
                'colormap': layer.colormap.name,
                'blending': layer.blending,
                'interpolation2d': layer.interpolation2d,
                'interpolation3d': layer.interpolation3d,
                'visible': layer.visible,
                'rendering': layer.rendering,
                'contrast_limits': layer.contrast_limits,
            }
            tupleOut[num][1].update(tmp)
            del viewer.layers[channel_name]
        else:
            # 如果找不到图层，使用默认参数
            tmp = {
                'opacity': 1.0,
                'gamma': 1.0,
                'colormap': 'gray',
                'blending': 'translucent',
                'interpolation2d': 'nearest',
                'interpolation3d': 'linear',
                'visible': True,
                'rendering': 'mip',
                'contrast_limits': [0, 255],
            }
            tupleOut[num][1].update(tmp)

    # 将新图层添加回 viewer
    for data, meta in tupleOut:
        print(f"Adding layer {meta['name']} with data shape {data[0].shape}")
        viewer.add_image(data, **meta)

    # 将 viewer 切换回原来的显示模式（如果需要）
    # viewer.dims.ndisplay = 3  # 如果您想在更新后切换回 3D 模式，可以取消注释

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return resolution_change