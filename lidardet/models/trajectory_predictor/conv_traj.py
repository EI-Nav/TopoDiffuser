import math
import torch
import torch.nn as nn
import numpy as np

from .conv_header import ConvHeader
from .backbone import Bottleneck, BackBone
from .predictor_base import PredictorBase
from ..registry import TRAJECTORY_PREDICTOR
# from .header import ConvHeader

@TRAJECTORY_PREDICTOR.register
class ConvTrajPredictor(PredictorBase):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        
        self.backbone = BackBone(Bottleneck, cfg.backbone)
        self.header = ConvHeader(cfg.header)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, batch_dict):
        x = batch_dict['lidar_bev']
        img_hmi = batch_dict['img_hmi']
        img_ins_past = batch_dict['img_ins_past']
        vla_tag = batch_dict['vla_tag']

        # print("x shape img_hmi shape", x.shape, img_hmi.shape)
        # angle_nomal = batch_dict['angle']
        # 首先将其扩展为32 x 1 x 1，这样每个原始元素就变成了一个独立的子张量
        # expanded_tensor = angle_nomal.view(img_hmi.shape[0], 1, 1)

        # 然后使用repeat方法来复制这些值以匹配300x400的尺寸
        # 注意，在每个维度上我们想要多少个副本。对于第一个维度（32），我们只需要1个副本，
        # 对于第二个维度（1），我们需要300个副本，对于第三个维度（1），我们需要400个副本。
        # result_angle = expanded_tensor.repeat(1, img_hmi.shape[2], img_hmi.shape[3]).view(img_hmi.shape[0], 1, img_hmi.shape[2], img_hmi.shape[3])
        # print("angle", angle_nomal.shape)
        # 维度扩展操作 (关键步骤)
        vla_tag = vla_tag.unsqueeze(-1).unsqueeze(-1)  # 形状变为 [8, 3, 1, 1]
        vla_tag = vla_tag.expand(-1, -1, 300, 400)    # 形状变为 [8, 3, 300, 400]
        x = torch.cat([x, img_hmi,img_ins_past], dim=1)
        # x = torch.cat([x, img_hmi,img_ins_past,vla_tag], dim=1)
        # x = torch.cat([x, img_hmi], dim=1)
        # print("x_last:",x.shape)

        batch_dict['input'] = x
        batch_dict = self.backbone(batch_dict)
        batch_dict,loss,tb_dict = self.header(batch_dict)

        if self.training:
            disp_dict = {}
            tb_dict = {
            'loss': loss.item(),
            **tb_dict
        }
            # loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts
