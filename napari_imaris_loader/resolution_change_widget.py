import napari
from magicgui import magic_factory
from napari_plugin_engine import napari_hook_implementation
from .reader import ims_reader
from typing import List
from napari.layers import Image
from napari.types import LayerDataTuple

@magic_factory(
    auto_call=False,
    call_button="Update",
    lowest_resolution_level={
        'min': 0,
        'max': 9,
        'tooltip': '''Important only for 3D rendering.
        Higher number is lower resolution.'''
    }
)
def resolution_change(
    viewer: napari.Viewer,
    lowest_resolution_level: int
):
    '''
    该面板提供了一个工具，用于在选择最低分辨率级别后重新加载 IMS 数据，
    该级别将包含在多分辨率系列中。较高的数字（即金字塔顶部）=较低的分辨率。

    这对于 3D 渲染很重要。如果您希望更高分辨率的 3D 渲染，可以选择较低的数字，更新视图，然后选择 3D 渲染。
    '''

    # 查找元数据中包含 'fileName' 的第一个图层
    ims_layer = None
    for layer in viewer.layers:
        if isinstance(layer, Image) and 'fileName' in layer.metadata:
            ims_layer = layer
            break

    if ims_layer is None:
        print("未找到元数据中包含 'fileName' 的 IMS 图层。")
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
        viewer.add_image(data, **meta)

    # 不需要返回任何内容


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return resolution_change