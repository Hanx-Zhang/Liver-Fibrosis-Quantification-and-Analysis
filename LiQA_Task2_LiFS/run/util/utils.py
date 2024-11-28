# -*- coding: utf-8 -*-

import SimpleITK as sitk
from monai.transforms import (
    KeepLargestConnectedComponent,
    ToNumpy,
    AsDiscrete,
    CastToType,
    AddChannel,
    SqueezeDim,
    ToTensor,
    EnsureChannelFirst,
)
import numpy as np
import torch


class InnerTransform(object):
    def __init__(self):
        self.ToNumpy = ToNumpy()
        self.AsDiscrete = AsDiscrete(threshold=0.5)
        self.ArgMax = AsDiscrete(argmax=True)
        self.KeepLargestConnectedComponent = KeepLargestConnectedComponent(applied_labels=1, connectivity=3)
        self.EnsureChannelFirst = EnsureChannelFirst()
        self.CastToNumpyUINT8 = CastToType(dtype=np.uint8)
        self.AddChannel = AddChannel()
        self.SqueezeDim = SqueezeDim()
        self.ToTensor = ToTensor(dtype=torch.float32)


InnerTransformer = InnerTransform()


def save_itk_with_backsampling(image, filename, origin, spacing, old_spacing, direction, old_size, islabel=True):
    if type(origin) != tuple:
        if type(origin) == list:
            origin = tuple(reversed(origin))
        else:
            origin = tuple(reversed(origin.tolist()))
    if type(spacing) != tuple:
        if type(spacing) == list:
            spacing = tuple(reversed(spacing))
        else:
            spacing = tuple(reversed(spacing.tolist()))
    if type(old_spacing) != tuple:
        if type(old_spacing) == list:
            old_spacing = tuple(reversed(old_spacing))
        else:
            old_spacing = tuple(reversed(old_spacing.tolist()))
    if type(direction) != tuple:
        if type(direction) == list:
            direction = tuple(reversed(direction))
        else:
            direction = tuple(reversed(direction.tolist()))
    if type(old_size) != tuple:
        if type(old_size) == list:
            old_size = tuple(reversed(old_size))
        else:
            old_size = tuple(reversed(old_size.tolist()))
    itkimage = sitk.GetImageFromArray(image, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    itkimage.SetDirection(direction)
    new_image_sitk, new_spacing_refine = ImageResample_to_newSize(itkimage, newSize=old_size, newSpacing=old_spacing, is_label=islabel)
    sitk.WriteImage(new_image_sitk, filename, True)


def ImageResample(sitk_image, new_spacing=[1.0, 1.0, 1.0], is_label=False):
    '''
    sitk_image:
    new_spacing: x,y,z
    is_label: if True, using Interpolator `sitk.sitkNearestNeighbor`
    '''

    size = np.array(sitk_image.GetSize())
    spacing = np.array(sitk_image.GetSpacing())
    new_spacing = np.array(new_spacing)
    new_size = size * spacing / new_spacing
    new_spacing_refine = size * spacing / new_size
    new_spacing_refine = [float(s) for s in new_spacing_refine]
    new_size = [int(s) for s in new_size]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetSize(new_size)
    resample.SetOutputSpacing(new_spacing_refine)

    if is_label:
        resample.SetOutputPixelType(sitk.sitkUInt8)
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetOutputPixelType(sitk.sitkFloat32)
        resample.SetInterpolator(sitk.sitkLinear)  # 1s

    newimage = resample.Execute(sitk_image)
    return newimage, new_spacing_refine, size


def ImageResample_to_newSize(sitk_image, newSize, newSpacing, is_label=False):
    '''
    sitk_image:
    new_spacing: x,y,z
    is_label: if True, using Interpolator `sitk.sitkNearestNeighbor`
    '''

    size = np.array(sitk_image.GetSize())
    spacing = np.array(sitk_image.GetSpacing())
    new_size = np.array(newSize, float)
    new_spacing = np.array(newSpacing, float)
    factor = size/new_size
    new_spacing_refine = spacing * factor
    new_size = new_size.astype(int)

    resample = sitk.ResampleImageFilter()
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetSize(new_size.tolist())
    resample.SetOutputSpacing(new_spacing)

    if is_label:
        resample.SetOutputPixelType(sitk.sitkUInt8)
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetOutputPixelType(sitk.sitkFloat32)
        resample.SetInterpolator(sitk.sitkLinear)

    newimage = resample.Execute(sitk_image)
    return newimage, new_spacing_refine


def load_itk_image_with_sampling(filename, spacing=[0.8, 0.8, 0.8]):
    itkimage = sitk.ReadImage(filename)
    new_image_sitk, new_spacing_refine, old_size = ImageResample(itkimage, new_spacing=spacing, is_label=False)
    numpyImage = sitk.GetArrayFromImage(new_image_sitk)  # z, y, x
    numpyOrigin = list(reversed(itkimage.GetOrigin()))
    numpySpacing = list(reversed(itkimage.GetSpacing()))
    numpyDirection = list(reversed(itkimage.GetDirection()))
    return new_image_sitk, numpyImage, numpyOrigin, numpySpacing, list(reversed(new_spacing_refine)), numpyDirection, \
           list(reversed(old_size))


def crop_image_via_box(image, box):
    return image[box[0, 0]:box[0, 1], box[1, 0]:box[1, 1], box[2, 0]:box[2, 1]]
