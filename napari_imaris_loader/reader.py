# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 20:40:39 2021

@author: AlanMWatson

Napari plugin for reading imaris files as a multiresolution series.

NOTE: Currently "File/Preferences/Render Images Asynchronously" must be turned on for this plugin to work

*** Issues remain with indexing and the shape of returned arrays.
1) It is unclear if there is an issue with how I am implementing slicing in the ims module
2) Different expectations from napari on the state of the data that is returned between the Image and Chunk_loader methods in ims module

** It appears that napari is only requesting 2D (YX) chunks from the loader during 2D rendering
which limits the utility of the async chunk_loader.

*Future implementation of caching in RAM and persistently on disk is planned via ims module - currently disabled
RAM Cache may be redundant to napari cache unless we can implement 3D chunk caching
Disk cache may allow for loaded chunks to be stored to SSD for rapid future retrieval
with options to maintain this cache persistently across sessions.
"""

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

    # Determine the channel axis
    if imsClass.TimePoints > 1:
        channel_axis = 1  # Assume time axis is first
    elif imsClass.Channels > 1:
        channel_axis = 0
    else:
        channel_axis = None

    meta['channel_axis'] = channel_axis

    # Ensure data shape is consistent and includes channel axis
    for idx in range(len(data)):
        # Expand dimensions if necessary to include channel axis
        if channel_axis is not None and data[idx].ndim < 4:
            data[idx] = data[idx][np.newaxis, ...]  # Add channel axis

    # Print data shapes for debugging
    print("Data shape after processing:", data[0].shape)

    # Set multiscale based on the number of resolution levels
    meta["multiscale"] = True if len(data) > 1 else False

    # Extract voxel spacing (scale)
    scale = imsClass.resolution
    # Adjust scale length to match data dimensions
    if data[0].ndim == len(scale) + 1:
        scale = (1,) + tuple(scale)  # Add scale for channel axis
    else:
        scale = tuple(scale)
    meta["scale"] = scale

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
                singleChannel.append(dd.take(cc, axis=channel_axis))
            channelData.append(singleChannel)

        del meta['channel_axis']  # Remove channel_axis from metadata

        # Create metadata for each channel
        metaData = []
        for cc in range(num_channels):
            singleChannelMeta = {
                'contrast_limits': meta['contrast_limits'],
                'multiscale': meta['multiscale'],
                'metadata': meta['metadata'],
                'scale': meta['scale'],
                'name': meta['name'][cc] if isinstance(meta['name'], list) else meta['name']
            }
            metaData.append(singleChannelMeta)

        # Prepare the final output
        finalOutput = []
        for dd, mm in zip(channelData, metaData):
            finalOutput.append((dd, mm))
        return finalOutput
    else:
        return [(data, meta)] if meta["multiscale"] else [(data[0], meta)]

@napari_hook_implementation
def napari_get_reader(path):
    if isinstance(path, str) and os.path.splitext(path)[1].lower() == '.ims':
        return ims_reader
