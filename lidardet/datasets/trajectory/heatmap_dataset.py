import cv2
import math
import time
import json
import pickle
import random
import numpy as np
from matplotlib import cm
from pathlib import Path
from torchvision.transforms import ToTensor

from .utils import interp_l2_distance, hausdorff_dist, HeatMapGenerator
from .utils import l2_distance_to_trajectory_set, angle_to_trajectory_set
from .utils import random_shift, random_drop, random_flip, random_rotate
from lidardet.datasets.base import PointCloudDataset
from lidardet.datasets.registry import DATASETS

from lidardet.ops.lidar_bev import bev
import torch

@DATASETS.register
class HeatMapDataset(PointCloudDataset):

    def __init__(self, cfg, logger=None):
        super(HeatMapDataset, self).__init__(cfg=cfg, logger=logger)
        # print("root_path: ", self.root_path)
        assert(self.root_path.exists())
        
        self.label_files = []
        for subdir in cfg.subset: 
            data_dir = self.root_path / subdir
            assert(data_dir.exists())
            fs = [p for p in data_dir.glob('*.json')]
            self.label_files.extend(fs)

        self.label_files.sort()
        
        self.res = cfg.voxel_size[0] # 0.16 m / pixel
        self.road_width = cfg.road_width # meters
        self.num_samples = cfg.num_samples
        
        xmin, ymin, zmin, xmax, ymax, zmax = cfg.point_cloud_range
        self.lidar_range_np = np.array([xmin, xmax, ymin, ymax, zmin, zmax])
        self.lidar_range = {'Left': xmin, 'Right': xmax, 'Front': ymax, 'Back': ymin, 'Bottom': zmin, 'Top': zmax}
        self.map_h = self.lidar_range['Front'] - self.lidar_range['Back']
        self.map_w = self.lidar_range['Right'] - self.lidar_range['Left']
        # print("map_h = ",self.map_h)
        # print("map_w = ",self.map_w)

        self.img_h = int(self.map_h / self.res) # 250
        self.img_w = int(self.map_w / self.res) # 400
        # print("img_h = ",self.img_h)
        # print("img_w = ",self.img_w)

        self.trajectory_colors = cm.get_cmap('viridis', 10)
        self.image_transform = ToTensor()
        self.heatmap_generator = HeatMapGenerator(self.img_w, self.img_h, 0.25)
      
        if self.logger:
            self.logger.info('Loading TrajectoryPrediction {} dataset with {} samples'.format(cfg.mode, len(self.label_files)))
       
    def __len__(self):
        return len(self.label_files)
    
    def img2car(self, img_points, normalized=True):
        """
        Convert coordinates from image space to ego-car
        """
        car_points = img_points.copy()
        if normalized:
            img_points[:,0] *= self.img_w
            img_points[:,1] *= self.img_h

        car_points[:,0] = img_points[:,0] * self.res + self.lidar_range['Left']
        car_points[:,1] = self.lidar_range['Front'] - img_points[:,1] * self.res
        return car_points

    def car2img(self, car_point, normalize=False):
        """
        Convert coordinates from ego-car to image space
        """
        if car_point.ndim == 2: # ndarray
            img_points = car_point.copy()
            img_points[:,0] = (car_point[:,0] - self.lidar_range['Left']) / self.res
            img_points[:,1] = (self.lidar_range['Front'] - car_point[:,1]) / self.res
            if normalize:
                img_points[:,0] /= self.img_w
                img_points[:,1] /= self.img_h
            return img_points

        cx, cy = car_point
        px = (cx - self.lidar_range['Left']) / self.res;
        py = (self.lidar_range['Front'] - cy) / self.res;
        if normalize:
            px /= self.img_w
            py /= self.img_h
        return np.array([int(px), int(py)], np.int32)
     
    def lidar_bev(self, points):
        m = self.cfg.lidar_intensity_max
        assert np.max(points[:,3]) <= m
        points[:,3] = points[:,3] / m
        
        if self.cfg.lidar_bev_type == 'height_map':
            bev_map = bev.height_map(points[:,:4], self.lidar_range_np, self.res)
        elif self.cfg.lidar_bev_type == 'rgb_map':
            bev_map = bev.rgb_map(points[:,:4], self.lidar_range_np, self.res)
        elif self.cfg.lidar_bev_type == 'rgb_traversability_map':
            bev_map = bev.rgb_traversability_map(points, self.lidar_range_np, self.cfg.sensor_height, self.res)

        bev_map = bev_map.reshape((self.img_h, self.img_w, -1)).copy()
        return bev_map

    def generate_road_mask(self, trajectory, rescale=1.0, road_width=2.0):
         
        thickness = int(rescale * road_width / self.res) # 16 pixels
        img = np.zeros((int(self.img_h * rescale), int(self.img_w * rescale), 1)).astype(np.uint8)
        for i in range(trajectory.shape[0] - 1):
            x1, y1 = self.car2img(trajectory[i][:2])
            x2, y2 = self.car2img(trajectory[i+1][:2])
                        
            #print(x1, y1, x2, y2)
            x1 = int(x1 * rescale)
            y1 = int(y1 * rescale)
            x2 = int(x2 * rescale)
            y2 = int(y2 * rescale)
            cv2.line(img, (x1, y1), (x2, y2), (255,255,255), thickness)

        return img
    def calculate_turn_angle_weights(self,waypoints):
        """
        计算轨迹点的转角权重（转角越大，权重越高）
        
        参数:
            waypoints (np.ndarray): 形状为 (N, 2) 的轨迹点数组，N为点数
        
        返回:
            np.ndarray: 形状为 (N,) 的权重数组，值在0~180之间
        """
        n = waypoints.shape[0]
        weights = np.zeros(n)
        for i in range(1, n-1):
        # 获取相邻三个点
            prev = waypoints[i-1]
            curr = waypoints[i]
            next_ = waypoints[i+1]
            
            # 计算向量
            vec_prev = curr - prev
            vec_next = next_ - curr
            
            # 计算夹角（弧度）
            dot_product = np.dot(vec_prev, vec_next)
            norm_prev = np.linalg.norm(vec_prev)
            norm_next = np.linalg.norm(vec_next)
            
            if norm_prev == 0 or norm_next == 0:
                angle = 0.0
            else:
                cos_theta = dot_product / (norm_prev * norm_next)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 避免数值误差
                angle = np.degrees(np.arccos(cos_theta))  # 转为角度
            
            weights[i] = angle  # 直接使用角度作为权重
        
        return max(weights)
    def generate_prediction_dicts(self, batch_dict, pred_dicts, output_path=None):
        result_dicts = {}
        
        bs = batch_dict['frame_id'].shape[0]
        num_modes = 1
        ADE = np.zeros((bs, num_modes)) # average displacement error
        HD = np.zeros((bs, num_modes)) # Hausdorff Distance
        FDE = np.zeros((bs,)) # final displacement error
        hit_rate = np.zeros((bs, num_modes))
        print('pred_traj: ',len(pred_dicts['pred_traj']))
        num_sample = self.num_samples
        print("num_sample: ", num_sample)
            # 用于存储所有帧的一致性分数
        vla_tag_con_scores = []
        for index, pred_seg in enumerate(pred_dicts['pred_seg']):
            heatmap = pred_dicts['pred_heatmap'][index]
            frame_id = batch_dict['frame_id'][index]
            start_index = index * num_sample
            end_index = start_index + num_sample
            pred_points = pred_dicts['pred_traj'][-1].detach().cpu().numpy()[start_index:end_index, :, :]
            # pred_points = pred_dicts['pred_traj'][index]

            traj_pred_all = pred_points
            # print(traj_pred_all.shape)
            # traj_pred_all = self.img2car(pred_points, normalized=True) # [num_points, 2]

            label_file = Path(self.label_files[frame_id])     
            # load json
            with open(str(label_file)) as f:
                data = json.load(f)
             
            lidar_path = str(label_file).replace('json', 'bin')
            points = np.fromfile(lidar_path, dtype=np.float32)
            points = points.reshape([-1, 4])

            # remove nan
            points = points[~np.isnan(points).any(axis=1)]
            lidar_bev = self.lidar_bev(points)
            
            if self.cfg.lidar_bev_type == 'height_map':
                img_bev = (lidar_bev[..., -1] * 255).astype(np.uint8)
                img_bev = img_bev[..., None]
            else: 
                img_bev = (lidar_bev[:,:,:3] * 255).astype(np.uint8)
       
            traj_hmi = np.array(data['trajectory_hmi'])
            traj_ins = np.array(data['trajectory_ins'])
            traj_ins_past = np.array(data['trajectory_ins_past'])
            vla_tag_true = self.determine_turn_direction(traj_ins)

            assert traj_ins_past.shape[0] >= 2
            dx = traj_ins_past[0,0] - traj_ins_past[1,0] 
            dy = traj_ins_past[0,1] - traj_ins_past[1,1]
            theta = np.arctan2(dx, dy)
            # constant velocity & yaw model
            traj_const = []
            r = 2.0 # radius
            for i in range(self.cfg.num_points_per_trajectory):
                x = r * np.sin(theta)
                y = r * np.cos(theta)
                if i > 0:
                    x1, y1 = traj_const[i - 1]
                    x += x1
                    y += y1
                traj_const.append(np.array([x, y]))
            traj_const = np.array(traj_const)

            assert traj_ins.shape[0] >= self.cfg.num_points_per_trajectory
            traj_ins = traj_ins[:self.cfg.num_points_per_trajectory,:2]
 
            assert traj_hmi.shape[0] >= self.cfg.num_points_per_trajectory
            traj_hmi = traj_hmi[:self.cfg.num_points_per_trajectory,:2]
           
            #traj_pred_all = traj_const 
            # 初始化变量来保存最优轨迹及其指标
            best_traj = None
            min_ADE = float('inf')
            min_FDE = float('inf')
            min_HD = float('inf')
            max_hit_rate = 0  # 我们希望找到的是1，即最佳命中率
            # 用于存储每个预测轨迹的 vla tag 是否一致
            vla_tag_consistency = []
            
            for pred_traj in traj_pred_all:
                
                # 计算当前预测轨迹与真实轨迹之间的插值L2距离
                _, _, metric = interp_l2_distance(pred_traj, traj_ins)
                
                # 计算Hausdorff距离
                hd = hausdorff_dist(pred_traj, traj_ins)
                
                # 判断是否找到了一个更优的轨迹
                should_update = False
                if metric['ADE'] < min_ADE or \
                metric['FDE'] < min_FDE or \
                hd < min_HD or \
                (metric['MAX'] < 2 and max_hit_rate == 0):  # 假设我们更看重hit_rate为1的情况
                    should_update = True
                    
                # 如果找到了更优的轨迹，则更新最优轨迹及其指标
                if should_update:
                    best_traj = pred_traj
                    min_ADE = metric['ADE']
                    min_FDE = metric['FDE']
                    min_HD = hd
                    max_hit_rate = 1 if metric['MAX'] < 2 else 0  # 根据'MAX'值更新hit_rate
                            # 获取预测轨迹的 vla tag
            vla_tag_pred = self.determine_turn_direction(pred_traj)
            
            # 比较预测轨迹和真实轨迹的 vla tag 是否一致
            is_consistent = vla_tag_pred == vla_tag_true
            vla_tag_consistency.append(is_consistent)
            
            # 计算当前帧的一致性分数（例如，平均一致性率）
            consistency_score = np.mean(vla_tag_consistency)
            vla_tag_con_scores.append(consistency_score)
            
            # 将最优轨迹的相关指标存入相应的数组中
            if best_traj is not None:
                ADE[index, 0] = min_ADE
                FDE[index] = min_FDE
                HD[index, 0] = min_HD
                hit_rate[index, 0] = max_hit_rate
            else:
                print("未找到符合条件的轨迹")
            # traj_pred_interp, traj_ins_interp, metric = interp_l2_distance(traj_pred_all[0], traj_ins)
            # FDE[index] = metric['FDE']
            
            # ADE[index, 0] = metric['ADE']
            # HD[index, 0] = hausdorff_dist(traj_pred_all[0], traj_ins)
            # hit_rate[index, 0] = 1 if metric['MAX'] < 2 else 0

            if output_path is not None:
                fname = label_file.stem
                save_dir = output_path / 'predictions' 
                save_dir.mkdir(parents=True, exist_ok=True)
                save_file = save_dir / ('%s.jpg' % fname)
                #print('Saving result {}'.format(save_file))
                
                pred_seg *= 255
                pred_seg = cv2.resize(pred_seg, (int(self.img_w), int(self.img_h)))
                img_seg = (img_bev + 0.3 * pred_seg[..., None]).astype(np.uint8)

                heatmap *= 255
                heatmap = cv2.resize(heatmap, (int(self.img_w), int(self.img_h)))
                img_hm = (img_bev * 0.3 + heatmap[..., None]).astype(np.uint8)
                
                img_traj = np.zeros((self.img_h, self.img_w, 3), np.uint8)
                img_traj += img_bev
                # print("traj_pred_all: ", traj_pred_all)
                img_traj = self.plot_trajectory_set([traj_ins, traj_pred_all, traj_hmi], 
                        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)], img=img_traj)
                if False:
                    # 确保输出目录存在
                    save_dir = output_path / 'separate_images'
                    save_dir.mkdir(parents=True, exist_ok=True)

                    # 分别保存三个图像（假设用frame_id作为文件名前缀）
                    # cv2.imwrite(str(save_dir / f'{fname}_bev.jpg'), img_bev)
                    # cv2.imwrite(str(save_dir / f'{fname}_hm.jpg'), img_hm)
                    # print(f'Saved images for {save_dir}')
                    
                    # # 在这里画每一个采样步的图
                    # step_imgs = []
                    # for i in range(len(pred_dicts['pred_traj'])):
                    #     step_img = np.zeros((self.img_h, self.img_w, 3), np.uint8)
                    #     step_img += img_bev 
                    #     pred_points_step = pred_dicts['pred_traj'][i].detach().cpu().numpy()[start_index:end_index, :, :]
                    #     step_img = self.plot_trajectory_set([traj_ins, pred_points_step, traj_hmi], 
                    #             colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)], img=step_img)
                    # #     cv2.putText(step_img, f"Time Step {i}", (10,30), 
                    # # cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    #     step_imgs.append(step_img)
                    # margin = 5                    # 图片间距（像素）
                    # background_color = (255, 255, 255)  # 背景颜色（BGR格式）
                    # cols = 5                      # 每行5张图片

                    # # 假设step_imgs已包含所有要拼接的图像（numpy数组格式）
                    # img_h, img_w = step_imgs[0].shape[:2]  # 获取图片高度和宽度
                    
                    # # 计算画布尺寸
                    # num_images = len(step_imgs)
                    # rows = math.ceil(num_images / cols)

                    # canvas_width = cols * img_w + (cols - 1) * margin
                    # canvas_height = rows * img_h + (rows - 1) * margin
                    
                    # # 创建白色背景画布（使用numpy数组操作更高效）
                    # canvas = np.full((canvas_height, canvas_width, 3), background_color, dtype=np.uint8)

                    # # 拼接图片
                    # for i, img in enumerate(step_imgs):
                    #     row = i // cols
                    #     col = i % cols
                        
                    #     # 计算粘贴位置
                    #     x = col * (img_w + margin)
                    #     y = row * (img_h + margin)
                        
                    #     # 确保图片尺寸一致
                    #     if img.shape[:2] == (img_h, img_w):
                    #         # 使用切片操作替换区域内容
                    #         canvas[y:y+img_h, x:x+img_w] = img
                    #     else:
                    #         print(f"Warning: Image {i} has unexpected size, skipped")

                    # # 保存结果
                    # cv2.imwrite(str(save_file), canvas)


                    # img_all = np.concatenate([img_traj, img_hm, img_seg], axis=1)
                    # cv2.imwrite(str(save_file), img_all)
                    if max_hit_rate > 0.93:
                        # print("angle: ", angle)
                        # save_dir = output_path / 'look' 
                        # save_dir.mkdir(parents=True, exist_ok=True)
                        save_file = save_dir / ('%s.jpg' % fname)
                        cv2.imwrite(str(save_file), img_traj)
                    # print("frame_id: ", frame_id) 
            
        result_dicts['ADE'] = ADE
        result_dicts['HD'] = HD
        result_dicts['FDE'] = FDE
        result_dicts['HitRate'] = hit_rate
        result_dicts['vla_tag_con_score'] = np.array(vla_tag_con_scores)
        return result_dicts

    def evaluation(self, result_dicts_list):
        self.logger.info('Evaluating on {}'.format(self.cfg.subset))
        for i, result_dicts in enumerate(result_dicts_list):
            HD_i = result_dicts['HD']
            ADE_i = result_dicts['ADE']
            FDE_i = result_dicts['FDE']
            hit_rate_i = result_dicts['HitRate']
            vla_tag_con_score_i = result_dicts['vla_tag_con_score']
            if i == 0:
                HD = HD_i
                ADE = ADE_i
                FDE = FDE_i
                hit_rate = hit_rate_i
                vla_tag_con_scores = vla_tag_con_score_i
            else:
                HD = np.concatenate((HD, HD_i), axis=0)
                ADE = np.concatenate((ADE, ADE_i), axis=0)
                FDE = np.concatenate((FDE, FDE_i), axis=0)
                hit_rate = np.concatenate((hit_rate, hit_rate_i), axis=0)
                vla_tag_con_scores = np.concatenate((vla_tag_con_scores, vla_tag_con_score_i), axis=0)
        
        self.logger.info('FDE =  {}'.format(np.mean(FDE)))
        for i in range(HD.shape[1]):
            self.logger.info('HD@{} = {}'.format(i, np.mean(HD[:,i])))

        for i in range(ADE.shape[1]):
            self.logger.info('minADE@{} = {}'.format(i, np.mean(ADE[:,i])))

        for i in range(hit_rate.shape[1]):
            self.logger.info('HitRate@{} = {}'.format(i, np.sum(hit_rate[:,i])/hit_rate.shape[0]))
        
        vla_tag_con_score_mean = np.mean(vla_tag_con_scores)
        self.logger.info('vla_tag_con_score_mean = {}'.format(vla_tag_con_score_mean))

    def draw_trajectory(self, img, trajectory, color, thickness=3):
            # 如果轨迹是一个元组，则转换为numpy数组
        # if isinstance(trajectory, tuple) or isinstance(trajectory, list):
        #     trajectory = np.array(trajectory)
        if trajectory.shape == (3,):  # 如果trajectory的形状为(3,)
            # print("Warning: Trajectory shape is unexpected (3,). Function will return without processing.")
            return  # 直接返回，不执行后续代码
        # hmi_start = (0,0)
        # hmi_end = (0,0)
        for i in range(trajectory.shape[0] - 1):
            # print("trajectory shape: ", trajectory.shape)
            x1, y1 = self.car2img(trajectory[i][:2])
            x2, y2 = self.car2img(trajectory[i+1][:2])
            
            if color[0] == 30: # hmi trajectory
                if i == 0 :
                    cv2.circle(img, (x1, y1), 2, color, thickness) # start point
                    hmi_start = (x1, y1)
                if i == trajectory.shape[0] - 2 :
                    cv2.circle(img, (x2, y2), 2, color, thickness) # end point
                    hmi_end = (x2, y2)


            #else:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
        
        # # calculate angle
        # if hmi_end == (0,0) and hmi_start == (0,0):
        #     return
        # tan_angle = (hmi_end[0] - hmi_start[0]) / (hmi_end[1] - hmi_start[1])
        # angleInRadians = np.arctan(tan_angle)
        # angleInDegrees = np.degrees(angleInRadians)
        # normal_angle = angleInDegrees/90.0 
        # # angleInDegrees = angleInRadians * (180.0 / 3.14);
    
        #                     # 文字内容
        # text = "angle: " + str(angleInDegrees)

        # # 设置字体
        # font = cv2.FONT_HERSHEY_SIMPLEX

        # # 设置字体大小
        # font_scale = 0.5

        # # 设置文字颜色 (B, G, R)
        # color = (0, 0, 255) # 红色

        # # 设置文字厚度
        # thickness = 2

        # # 获取文字尺寸以便计算放置位置，避免文字超出图像边界
        # text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

        # # 计算文字起始位置，这里我们让文字居中显示
        # text_x = (img.shape[1] - text_size[0]) // 2
        # text_y = (img.shape[0] + text_size[1]) // 2

        #             # 在图像上添加文字
        # cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA) 


    def plot_trajectory_set(self, trajectory_set, colors=[], img=None, thickness=3):
        if img is None:
            img = np.zeros((self.img_h, self.img_w, 3)).astype(np.uint8)

        for t, trajectory in enumerate(trajectory_set):
            
            if len(colors) > 0:
                color = colors[t]
            else:
                color = [int(255 * c) for c in self.trajectory_colors(1.0 * t / len(trajectory_set))]
            if (t==1 and len(trajectory_set) == 3) : # draw predicted trajectory
                for i in range(trajectory.shape[0]):
                    self.draw_trajectory(img, trajectory[i], color, thickness)
            else:
                self.draw_trajectory(img, trajectory, color, thickness)


        return img

    def _get_turn_label(self, result):
        """将判断结果转换为文字标签"""
        if isinstance(result, list):
            if result == [1, 0, 0]:
                return "左转"
            elif result == [0, 1, 0]:
                return "直行"
            elif result == [0, 0, 1]:
                return "右转"
        return "未知"  # 处理无法判断的情况
    def determine_turn_direction(self,trajectory, threshold_ratio=0.07):
        if len(trajectory) < 3:
            return "轨迹点不足，无法判断"
        
        sum_cross = 0.0
        total_length = 0.0
        
        for i in range(len(trajectory) - 2):
            p0 = trajectory[i]
            p1 = trajectory[i+1]
            p2 = trajectory[i+2]
            
            # 计算向量和长度
            v1x, v1y = p1[0]-p0[0], p1[1]-p0[1]
            v2x, v2y = p2[0]-p1[0], p2[1]-p1[1]
            segment_length = max(abs(v1x)+abs(v1y), 1e-5)  # 防止除以零
            
            # 累加归一化后的叉积
            sum_cross += (v1x * v2y - v1y * v2x) / segment_length
            total_length += segment_length

        # 动态阈值 = 路径总长 × 阈值比例系数
        dynamic_threshold = total_length * threshold_ratio
        
        # One-hot编码生成
        if abs(sum_cross) < dynamic_threshold:
            return [0, 1, 0]  # 直行
        elif sum_cross > 0:
            return [1, 0, 0]  # 左转
        else:
            return [0, 0, 1]  # 右转
    def get_angle(self,trajectory):
        angle = self.calculate_turn_angle_weights(trajectory)
        x1, y1 = self.car2img(trajectory[0][:2])
        hmi_start = (x1, y1)
        x2, y2 = self.car2img(trajectory[trajectory.shape[0] - 1][:2])
        hmi_end = (x2, y2)
                # calculate angle
        if (hmi_end[1] - hmi_start[1]) == 0:
            if hmi_end[0] - hmi_start[0] >0 :
                return 1.0
            else:
                return -1.0

        tan_angle = (hmi_end[0] - hmi_start[0]) / (hmi_end[1] - hmi_start[1])
        angleInRadians = np.arctan(tan_angle)
        angleInDegrees = np.degrees(angleInRadians)
        angleInDegrees = max(abs(angleInDegrees), angle)
        # angleInDegrees = angleInDegrees/90.0 
        # angleInDegrees = angleInRadians * (180.0 / 3.14);
                            # 文字内容
 
        return angleInDegrees
    def plot_angle(self,angleInDegrees,img):
        text = "angle: " + str(angleInDegrees)

        # 设置字体
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 设置字体大小
        font_scale = 0.5

        # 设置文字颜色 (B, G, R)
        color = (0, 0, 255) # 红色

        # 设置文字厚度
        thickness = 2

        # 获取文字尺寸以便计算放置位置，避免文字超出图像边界
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
        # # 计算文字起始位置，这里我们让文字居中显示
        text_x = (img.shape[1] - text_size[0]) // 2
        text_y = (img.shape[0] + text_size[1]) // 2

                    # 在图像上添加文字
        cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA) 
    def __getitem__(self, idx):
        label_file = Path(self.label_files[idx]) 
        self.curr_file = label_file
        # load json
        with open(str(label_file)) as f:
            data = json.load(f)

        lidar_path = str(label_file).replace('json', 'bin')
        points = np.fromfile(lidar_path, dtype=np.float32)
        points = points.reshape([-1, 4])
        
        # remove nan
        points = points[~np.isnan(points).any(axis=1)]
 
        traj_ins = np.array(data['trajectory_ins'])
        traj_hmi = np.array(data['trajectory_hmi'])
        
        traj_ins_past = np.array(data['trajectory_ins_past'])
        traj_hmi_past = np.array(data['trajectory_hmi_past'])
        
        if self.cfg.mode == 'train':
            # random drop points
            if self.cfg.random_drop:
                points = random_drop(points)

            # random rotate
            if self.cfg.random_rotate:
                points, traj_list = random_rotate(points, [traj_ins, traj_ins_past, traj_hmi, traj_hmi_past])
                traj_ins, traj_ins_past, traj_hmi, traj_hmi_past = traj_list

            # random flip
            if self.cfg.random_flip:
                points, traj_list = random_flip(points, [traj_ins, traj_ins_past, traj_hmi, traj_hmi_past])
                traj_ins, traj_ins_past, traj_hmi, traj_hmi_past = traj_list

            # random shift hmi trajectory
            if self.cfg.random_shift:
                traj_hmi, taj_hmi_past = random_shift([traj_hmi, traj_hmi_past], self.res, self.cfg.get('max_random_shift', 1.0))
        
        # set origin on the ground 
        points[:,2] += self.cfg.sensor_height

        traj_ins_all = np.vstack((traj_ins_past[::-1], traj_ins)) 
        traj_hmi_all = np.vstack((traj_hmi_past[::-1], traj_hmi)) 
        traj_ins_past = traj_ins_past[::-1]

        img_ins = self.generate_road_mask(traj_ins_all, rescale=0.25, road_width=self.road_width) 
        img_hmi = self.generate_road_mask(traj_hmi_all, rescale=1.0, road_width=self.road_width) 
        img_ins_past = self.generate_road_mask(traj_ins_past, rescale=1.0, road_width=self.road_width)
        
        img_ins_save = img_ins.copy()

        lidar_bev = self.lidar_bev(points)
 
        img_ins = img_ins[...,0]
        img_ins = img_ins / 255.0
        
        img_hmi_save = img_hmi.copy()
        img_hmi = self.image_transform(img_hmi)
        # print("img_hmi:", img_hmi.shape)

        img_ins_past_save = img_ins_past.copy()
        img_ins_past = self.image_transform(img_ins_past)
        # img_ins_past = img_ins_past.squeeze(dim=0)
        # print("img_ins_past:", img_ins_past.shape)
        
        assert traj_ins.shape[0] >= self.cfg.num_points_per_trajectory
        traj_ins = traj_ins[:self.cfg.num_points_per_trajectory,:2]
        
        assert traj_hmi.shape[0] >= self.cfg.num_points_per_trajectory
        traj_hmi = traj_hmi[:self.cfg.num_points_per_trajectory,:2]
        
        n_past = min(traj_ins_past.shape[0], self.cfg.num_points_per_trajectory)
        traj_hist = np.zeros_like(traj_ins)
        traj_hist[:n_past] = traj_ins_past[:n_past,:2]
        
        traj_ins_pixel_norm = self.car2img(traj_ins, normalize=True)
        heatmap = self.heatmap_generator(traj_ins_pixel_norm.copy())
        vla_tag = self.determine_turn_direction(traj_ins)
        turn_label = self._get_turn_label(vla_tag)

        """保存转向结果到JSON文件"""

        fname = label_file.stem
        output_path = Path(self.cfg.get('output_path', 'output'))
        save_dir = output_path / 'separate_images_09'
        save_dir.mkdir(parents=True, exist_ok=True)
        json_path = save_dir / ('%s.json' % fname)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(turn_label, f, ensure_ascii=False, indent=4)
        print(f"转向结果已保存至 {json_path}")
        
        if False:
            
            if self.cfg.lidar_bev_type == 'rgb_traversability_map':
                obstacle_map1 = lidar_bev[...,-2] # infered traversability map
                img_obstacle1 = np.zeros((self.img_h, self.img_w, 3)).astype(np.uint8)
                img_obstacle1[obstacle_map1 == 0, 0] = 255 
                img_obstacle1[obstacle_map1 == 2, 1] = 255 
                img_obstacle1[obstacle_map1 == 1, 2] = 255 
                
                obstacle_map2 = lidar_bev[...,-1] # original
                img_obstacle2 = np.zeros((self.img_h, self.img_w, 3)).astype(np.uint8)
                img_obstacle2[obstacle_map2 == 0, 0] = 255 
                img_obstacle2[obstacle_map2 == 2, 1] = 255 
                img_obstacle2[obstacle_map2 == 1, 2] = 255 

                cv2.imshow('obstacle_map', np.concatenate([img_obstacle2, img_obstacle1], axis=1))
       
            img_bev = (lidar_bev[:,:,:3] * 255).astype('uint8')
            # img_bev = self.plot_trajectory_set(trajectory_set=[traj_ins_all, traj_hmi_all], img=img_bev, thickness=3)
            img_bev = self.plot_trajectory_set(trajectory_set=[traj_ins, traj_hmi], img=img_bev, thickness=3)
                     
            fname = label_file.stem
            output_path = Path(self.cfg.get('output_path', 'output'))
            save_dir = output_path / 'separate_images'
            save_dir.mkdir(parents=True, exist_ok=True)
            save_file = save_dir / ('%s.jpg' % fname)
            save_file_hmi = save_dir / ('%s_hmi.jpg' % fname)
            
            # if(int(fname) == 291):
            #     print("save_file: ", save_file)
            #     print(traj_hmi)
            
            print('Saving result {}'.format(save_file))
            # cv2.imwrite(str(save_file), img_ins_save)
            # cv2.imwrite(str(save_file_hmi), img_hmi_save)
            # heatmap_plot = heatmap.sum(axis=1)
            # heatmap_plot *= 255
            # heatmap_plot = cv2.resize(heatmap_plot, (int(self.img_w), int(self.img_h)))
            # img_hm_plot = (img_bev * 0.3 + heatmap_plot[..., None]).astype(np.uint8)
            # save_file_hm = save_dir / ('%s_hm.jpg' % fname)
            # cv2.imwrite(str(save_file_hm), img_hm_plot)
            # print('Saving result {}'.format(save_file))
            # cv2.imshow("lidar_bev", img_bev)
            # cv2.waitKey(0)
            # exit(0)

        if False:
            cv2.imshow("heatmap", np.sum(heatmap, axis=0))
            cv2.waitKey(0)
            exit(0)
        # angle = self.get_angle(traj_ins)

        data_dict = {
                'img_hmi': img_hmi,
                'img_ins': img_ins,
                'frame_id': idx,
                'traj_ins': traj_ins,
                'traj_ins_pixel_norm': traj_ins_pixel_norm,
                'traj_hmi': traj_hmi,
                'traj_hist': traj_hist,
                'heatmap': heatmap,
                'img_ins_past':img_ins_past,
                'vla_tag': vla_tag
                # 'angle': angle
                }
        
        if self.cfg.use_lidar_points:
            data_dict['points'] = points

        if self.cfg.lidar_bev_type == 'rgb_traversability_map':
            lidar_bev[...,-2] -= 1 # convert to range [-1, 0, 1]
            data_dict['lidar_bev'] = lidar_bev[:,:,:-1].transpose(2, 0, 1)
            obstacle_map = lidar_bev[...,-2].copy() 
            obstacle_map[obstacle_map != 0] = 1;
            # rescale
            scale = 0.25
            obstacle_map = cv2.resize(obstacle_map, (int(self.img_w * scale), int(self.img_h * scale)))
            data_dict['obstacle_map'] = obstacle_map

            #cv2.imshow("obstacle_map", (obstacle_map * 255).astype(np.uint8))
            #cv2.waitKey(0)
            #exit(0)
        else:
            # H, W, C -> C, H, W
            data_dict['lidar_bev'] = lidar_bev[:,:,:3].transpose(2, 0, 1)

        if self.cfg.get('pre_processor', None):
            data_dict = self.pre_process(data_dict)

        return data_dict  
