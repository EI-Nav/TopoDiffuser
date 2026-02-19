import math
import numpy as np
import torch
from torch import nn

from .backbone import conv3x3
from .transformer import Transformer

def print_color(text, color):
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'reset': '\033[0m',  # Reset to default color
    }
    if color not in colors:
        print(text)
    else:
        print(f"{colors[color]}{text}{colors['reset']}")

class ConvHeader(nn.Module):

    def __init__(self, cfg):
        super(ConvHeader, self).__init__()
        # self.cfg = cfg
        self.use_bn = cfg.use_bn
        self.use_transformer = cfg.use_transformer
        self.use_position_encoder = cfg.transformer.use_position_encoder
        self.use_road_seg = cfg.use_road_seg
        self.use_diffusion_policy = cfg.use_diffusion_policy

        self.weight_loss_heatmap = cfg.weight_loss_heatmap
        self.weight_loss_waypoint = cfg.weight_loss_waypoint
        self.weight_loss_road = cfg.weight_loss_road

        bias = not self.use_bn
        self.conv1 = conv3x3(96, 96, bias=bias)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = conv3x3(96, 96, bias=bias)
        self.bn2 = nn.BatchNorm2d(96)
        self.conv3 = conv3x3(96, 96, bias=bias)
        self.bn3 = nn.BatchNorm2d(96)
        self.conv4 = conv3x3(96, 96, bias=bias)
        self.bn4 = nn.BatchNorm2d(96)
        
        self.heatmap_head = conv3x3(96, cfg.num_points_per_trajectory, bias=True)
        self.softmax = nn.Softmax(dim=-1)

        npoints = cfg.num_points_per_trajectory
        if self.use_road_seg:
            self.road_head = conv3x3(96, 1, bias=True)
            self.conv7 = nn.Conv2d(1, npoints, 3, 2, padding=1, bias=True)

        if cfg.use_transformer:
            self.conv5 = nn.Conv2d(96, npoints, 3, 2, padding=1, bias=True)
            # depth-wise convolution
            if cfg.transformer.use_position_encoder:
                self.conv6 = nn.Conv2d(npoints, npoints, 3, 2, groups=npoints, padding=1, bias=True)
            self.waypoint_predictor = Transformer(cfg.transformer) 

        self.road_loss_func = nn.BCELoss().cuda()
        self.heatmap_loss_func = nn.MSELoss().cuda()

    def spatial_softmax(self, x):
        b, c, h, w = x.shape
        x = self.softmax(x.view(b, c, -1))
        x = x.view(b, c, h, w)

        argmax = torch.argmax(x.view(b * c, -1), dim=1)
        argmax_x, argmax_y = torch.remainder(argmax, w).float(), torch.floor(torch.div(argmax.float(), float(w)))
        argmax_x = argmax_x.view((b, c, -1)) / float(w)
        argmax_y = argmax_y.view((b, c, -1)) / float(h)
        pos_pred = torch.cat([argmax_x, argmax_y], dim=2)

        return x, pos_pred

    def forward(self, batch_dict):
        feature_map = batch_dict['seg_features'] # [32, 96, 75, 100]
        print_color("feature_map.shape: {}".format(feature_map.shape), 'green')

        x = self.conv1(feature_map)
        if self.use_bn:
            x = self.bn1(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.conv3(x)
        if self.use_bn:
            x = self.bn3(x)
        x = self.conv4(x)
        if self.use_bn:
            x = self.bn4(x)
        # print_color("x.shape: {}".format(x.shape), 'green') # [32, 96, 75, 100]
        heatmap = self.heatmap_head(x) # [b, num_points_per_trajectory, 75, 100] [32, 10, 75, 100]
        heatmap, pred_pos = self.spatial_softmax(heatmap)

        if self.use_road_seg:
            road = torch.sigmoid(self.road_head(x)) # [b, 1, 75, 100]
            print_color("road.shape: {}".format(road.shape), 'green')
            batch_dict['pred_seg'] = road.squeeze(1)
        else:
            batch_dict['pred_seg'] = heatmap.sum(dim=1)

        batch_dict['pred_heatmap'] = heatmap.sum(dim=1)
        if self.use_transformer:
            waypoints_feature = self.conv5(feature_map) # 直接从卷积层获得的特征
            b, c, h, w = waypoints_feature.shape
            if self.use_position_encoder:
                x2 = self.conv6(heatmap) # heatmap特征
                batch_dict['pos_feature'] = x2.view(b, c, -1)

            if self.use_road_seg:
                x3 = self.conv7(road) # 道路分割特征
                # waypoints_feature += x3
                waypoints_feature = x3
                print_color("x3.shape: {}".format(x3.shape), 'green')
                print_color("waypoints_feature.shape: {}".format(waypoints_feature.shape), 'green')

            waypoints_feature = waypoints_feature.view(b, c, -1)

            batch_dict['waypoints_feature'] = waypoints_feature
            batch_dict,loss_waypoint,loss_last = self.waypoint_predictor(batch_dict)
            loss_waypoint = loss_waypoint * self.weight_loss_waypoint 
            
            tb_dict = {'loss_waypoint': loss_waypoint}
            
            tb_dict = {'loss_last': loss_last}


        else:
            batch_dict['waypoints_pred'] = pred_pos
        loss = 0
        if self.training:

            if self.use_road_seg:
                road_pred = road.squeeze(1)
                self.road_target = batch_dict['img_ins']
                loss_road = self.road_loss_func(road_pred, self.road_target)
                loss_road *= self.weight_loss_road 
                tb_dict['loss_road'] = loss_road

            # mean_value = torch.mean(heatmap)
            # print("Mean:", mean_value.item())
            # mean_value = torch.mean(batch_dict['heatmap'])
            # print("Mean pre:", mean_value.item())
            
            
            heatmap_pred = heatmap.detach()
            self.heatmap_targets = batch_dict['heatmap']
            loss_heatmap = self.heatmap_loss_func(heatmap_pred, self.heatmap_targets)
            # print_color("loss_heatmap: {}".format(loss_heatmap), 'green')
            loss_heatmap *= self.weight_loss_heatmap
            tb_dict['loss_heatmap'] = loss_heatmap
            loss = loss_heatmap + loss_waypoint + loss_road + loss_last
        return batch_dict,loss, tb_dict
    
    # def get_loss(self):

    #     loss_heatmap = self.loss_heatmap
    #     loss_heatmap *= self.weight_loss_heatmap
    #     loss = loss_heatmap
    #     tb_dict = {'loss_heatmap': loss_heatmap}

    #     if self.use_transformer:
    #         loss_waypoint = self.waypoint_predictor.get_loss()
    #         loss_waypoint *= self.weight_loss_waypoint
    #         loss += loss_waypoint
    #         tb_dict['loss_waypoint'] = loss_waypoint

    #     if self.use_road_seg:
    #         loss_road = self.loss_road
    #         loss_road *= self.weight_loss_road 
    #         loss += loss_road
    #         tb_dict['loss_road'] = loss_road

    #     return loss, tb_dict

    def get_prediction(self, batch_dict):
        # pred_points = batch_dict['waypoints_pred']
        pred_points = batch_dict['intermediate_samples']
        return pred_points
    
    def parse_predicted_waypoints(self, prediction):
        assert isinstance(prediction, torch.Tensor)
        predicted_points = []
        prediction = prediction.detach().cpu().numpy()
        c, h, w = prediction.shape[0:3]
        num_points = c
        assert num_points == self.cfg.num_points_per_trajectory
        for n in range(num_points):
            tmp = prediction[n]
            y, x = np.unravel_index(np.argmax(tmp), tmp.shape)
            point = np.array([x, y])
            predicted_points.append(point)
        return np.array(predicted_points).astype(np.float32)
