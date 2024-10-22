import os
import numpy as np
import dask.array as da
from imaris_ims_file_reader.ims import ims

from napari_plugin_engine import napari_hook_implementation

def ims_reader(path, resLevel='max', colorsIndependant=False, preCache=False):
    # Ensure NAPARI_ASYNC is enabled for asynchronous loading
    os.environ["NAPARI_ASYNC"] = "1"

    # Load the ims file
    squeeze_output = False  # Do not squeeze to preserve dimensions
    imsClass = ims(path, squeeze_output=squeeze_output)

    # Attempt to extract contrast limits from the lowest resolution level
    try:
        # Extract minimum resolution level and calculate contrast limits
        minResolutionLevel = imsClass[imsClass.ResolutionLevels - 1, :, :, :, :, :]
        minContrast = minResolutionLevel[minResolutionLevel > 0].min()
        maxContrast = minResolutionLevel.max()
        contrastLimits = [minContrast, maxContrast]
    except Exception:
        # Fallback contrast limits based on data type
        if imsClass.dtype == np.dtype('uint16'):
            contrastLimits = [0, 65535]
        elif imsClass.dtype == np.dtype('uint8'):
            contrastLimits = [0, 255]
        else:
            contrastLimits = [0, 1]

    # Prepare channel names
    channelNames = []
    for cc in range(imsClass.Channels):
        channelNames.append(f'Channel {cc}')
    if len(channelNames) == 1:
        channelNames = channelNames[0]

    # Load data for each resolution level
    data = []
    for rr in range(imsClass.ResolutionLevels):
        print(f'Loading resolution level {rr}')
        data.append(ims(path, ResolutionLevelLock=rr, cache_location=imsClass.cache_location, squeeze_output=squeeze_output))

    # Convert data to Dask arrays with appropriate chunking
    for idx, _ in enumerate(data):
        data[idx] = da.from_array(
            data[idx],
            chunks=data[idx].chunks,
            fancy=False
        )

    # Base metadata that applies to all scenarios
    meta = {
        "contrast_limits": contrastLimits,
        "name": channelNames,
        "metadata": {
            'fileName': imsClass.filePathComplete,
            'resolutionLevels': imsClass.ResolutionLevels
        }
    }

    # Determine the channel axis based on data shape and imsClass.Channels
    channel_axis = None
    for idx, dim in enumerate(data[0].shape):
        if dim == imsClass.Channels and imsClass.Channels > 1:
            channel_axis = idx
            break

    meta['channel_axis'] = channel_axis

    # Ensure data shape is consistent and includes channel axis
    for idx in range(len(data)):
        # Expand dimensions if necessary to include channel axis
        if channel_axis is not None and data[idx].ndim <= channel_axis:
            data[idx] = np.expand_dims(data[idx], axis=channel_axis)

    # Set multiscale based on the number of resolution levels
    meta["multiscale"] = True if len(data) > 1 else False

    # Optionally limit the number of resolution levels
    if isinstance(resLevel, int):
        if resLevel + 1 > len(data):
            raise ValueError(f'Selected resolution level is too high: Options are between 0 and {imsClass.ResolutionLevels - 1}')
        data = data[:resLevel + 1]

    # Handle independent colors (channels)
    if colorsIndependant and channel_axis is not None:
        channelData = []
        num_channels = data[0].shape[channel_axis]
        for cc in range(num_channels):
            singleChannel = []
            for dd in data:
                # Use slicing to extract the channel
                indexer = [slice(None)] * dd.ndim
                indexer[channel_axis] = cc
                singleChannel.append(dd[tuple(indexer)])
            channelData.append(singleChannel)

        del meta['channel_axis']  # Remove channel_axis from metadata

        # Create metadata for each channel
        metaData = []
        for cc in range(num_channels):
            # Adjust scale for each channel
            data_shape = channelData[cc][0].shape
            extra_dims = len(data_shape) - len(imsClass.resolution)
            singleChannelScale = (1,) * extra_dims + tuple(imsClass.resolution)

            singleChannelMeta = {
                'contrast_limits': meta['contrast_limits'],
                'multiscale': meta['multiscale'],
                'metadata': meta['metadata'],
                'scale': singleChannelScale,
                'name': meta['name'][cc] if isinstance(meta['name'], list) else meta['name']
            }
            metaData.append(singleChannelMeta)

        # Prepare the final output
        finalOutput = []
        for dd, mm in zip(channelData, metaData):
            finalOutput.append((dd, mm))
        return finalOutput
    else:
        # Adjust scale for combined data
        data_shape = data[0].shape
        extra_dims = len(data_shape) - len(imsClass.resolution)
        scale = (1,) * extra_dims + tuple(imsClass.resolution)

        # If channel_axis is specified, adjust scale for per-channel data
        if channel_axis is not None:
            # After splitting, data per channel will have one less dimension
            per_channel_shape = data_shape[:channel_axis] + data_shape[channel_axis + 1:]
            per_channel_extra_dims = len(per_channel_shape) - len(imsClass.resolution)
            per_channel_scale = (1,) * per_channel_extra_dims + tuple(imsClass.resolution)
            meta["scale"] = per_channel_scale
        else:
            meta["scale"] = scale

        return [(data, meta)] if meta["multiscale"] else [(data[0], meta)]


@napari_hook_implementation
def napari_get_reader(path):
    if isinstance(path, str) and os.path.splitext(path)[1].lower() == '.ims':
        return ims_reader
