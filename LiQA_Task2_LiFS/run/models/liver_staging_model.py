# -*- coding: utf-8 -*-

from networks.resnet_groupnorm_siamese import resnet34
from configs.liver_staging_config import config
from util.utils import InnerTransformer

import torch
import numpy as np
import torch.nn.functional as F

class LiverStagingModel(object):
    def __init__(self):
        self.config = config
        self.device = self.config['device']
        self.net = resnet34(
                shortcut_type='B',
                num_seg_classes=1)

        self.net = self.net.to(self.device)
        self.net = torch.nn.DataParallel(self.net, device_ids=list(
            range(torch.cuda.device_count()))).to(self.device)
        checkpoint = torch.load(self.config['weight_path'], map_location='cpu')
        self.net.load_state_dict(checkpoint['state_dict'])

        self.lung_boundingbox = None

    @torch.no_grad()
    def predict(self, image: np.ndarray):
        self.net.eval()
        image = np.transpose(image, (3, 1, 2, 0))
        image = InnerTransformer.ToTensor(image)
        image = InnerTransformer.AddChannel(image)
        image = image.to(self.device)

        predictor = self.net

        pred = predictor(image)

        pred = F.softmax(InnerTransformer.SqueezeDim(pred))
        pred = pred.cpu().numpy()

        return pred


