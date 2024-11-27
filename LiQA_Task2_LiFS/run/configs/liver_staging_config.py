# -*- coding: utf-8 -*-

config = {
    "in_channels": 2,
    "out_channels": 6,
    "device": "cuda:0",
    "weight_path": './checkpoints/CE_Res34_all.pth',
    "roi_size": (160, 160, 96),
    "sw_batch_size": 1,
    "overlap": 0.50,
    "KeepLargestConnectedComponent": True,
    "CalculateLungBoundingbox": True,
    "margin_lung_boundingbox": 5,
}
