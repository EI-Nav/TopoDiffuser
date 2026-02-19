import torch
from torch import nn
from copy import deepcopy
from typing import Optional 
import torch.nn.functional as F
from torch import nn, Tensor
# import hydra
import einops
import numpy as np

from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
# from diffusers.schedulers.scheduling_ddim import DDIMScheduler
# from diffusers import DPMSolverMultistepScheduler
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusers.training_utils import EMAModel
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

ACTION_STATS = {}
def get_delta(actions):
    actions = actions.cpu().detach().numpy()
    # append zeros to first action
    ex_actions = np.concatenate([np.zeros((actions.shape[0],1,actions.shape[-1])), actions], axis=1)
    delta = ex_actions[:,1:] - ex_actions[:,:-1]
    delta = torch.from_numpy(delta).float()
    return delta

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    data = data.cpu().detach().numpy()
    # print_color(f"data shape: {data.shape}", 'red')
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    data =data.cpu().detach().numpy() 
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    ndata = torch.from_numpy(ndata).float()
    return ndata

def unnormalize_data(ndata, stats):
    ndata = ndata.cpu().detach().numpy()  
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    data = np.cumsum(data, axis=1)
    data = torch.from_numpy(data).float()
    return data

class Transformer(nn.Module):
    def __init__(self, cfg):
        super(Transformer, self).__init__()
        
        # self.cfg = cfg
        input_dim = cfg.input_dim
        d_model = cfg.d_model
        nhead = cfg.nhead
        nhid = cfg.dim_feedforward
        nlayers = cfg.num_layers
        dropout = cfg.dropout
        self.use_position_encoder = cfg.use_position_encoder
        self.use_transformer_encoder = cfg.use_transformer_encoder
        self.use_transformer_decoder = cfg.use_transformer_decoder
        self.device = torch.device(cfg.device)
        self.num_train_timesteps = cfg.num_train_timesteps
        self.beta_start = cfg.beta_start
        self.beta_end = cfg.beta_end
        self.beta_schedule = cfg.beta_schedule

        self.down_dims = cfg.down_dims
        self.cond_predict_scale = cfg.cond_predict_scale
        self.global_cond_dim = cfg.global_cond_dim
        self.num_samples = cfg.num_samples
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=2,
            global_cond_dim=self.global_cond_dim,
            down_dims=self.down_dims,
            cond_predict_scale=self.cond_predict_scale,
            # diffusion_step_embed_dim = 512
        ).to(self.device) # 要将模型转移到cuda上  
        obs_as_global_cond = True
        obs_feature_dim = 4
        n_obs_steps = 10
        
        for key in cfg.action_stats:
            ACTION_STATS[key] = np.array(cfg.action_stats[key])

        self.feature_encoder = Encoder(input_dim, d_model, cfg.feature_encoder_layers)
        if self.use_position_encoder:
            self.pos_encoder = Encoder(input_dim, d_model, cfg.position_encoder_layers)

        if self.use_transformer_encoder:
            encoder_layer = TransformerEncoderLayer(cfg)
            self.transformer_encoder = TransformerEnocder(encoder_layer, cfg)
            self._reset_parameters(self.transformer_encoder)

        if self.use_transformer_decoder:
            self.query_embed = nn.Embedding(cfg.num_points_per_trajectory, d_model)
            decoder_layer = TransformerDecoderLayer(cfg)
            self.transformer_decoder = TransformerDecoder(decoder_layer, cfg)
            self._reset_parameters(self.transformer_decoder)
        
        self.proj = nn.Conv1d(cfg.d_model, 2, kernel_size=1, bias=True)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=cfg.num_train_timesteps,beta_start=cfg.beta_start,beta_end=cfg.beta_end,beta_schedule=cfg.beta_schedule)

        # self.noise_scheduler = DPMSolverMultistepScheduler(
        #     num_train_timesteps=cfg.num_train_timesteps,  # 需与训练时的总步数一致
        #     beta_start=cfg.beta_start,
        #     beta_end=cfg.beta_end,
        #     beta_schedule=cfg.beta_schedule,
        #     algorithm_type="dpmsolver++",
        #     solver_order=2,
        #     thresholding=False
        # )

        
        # # 初始化 DDIMScheduler
        # self.noise_scheduler = DDIMScheduler(
        #     num_train_timesteps=cfg.num_train_timesteps,
        #     beta_start=cfg.beta_start,
        #     beta_end=cfg.beta_end,
        #     beta_schedule=cfg.beta_schedule,
        #     # 根据需要添加其他参数
        #     # clip_sample=True,  # 如果你想限制样本范围，可以设为True
        #     # set_alpha_to_one=True  # 影响最终步的alpha值，如果你想设置为1，可以设为True
        # )
        

        if cfg.get('loss_type', 'mse') == 'smooth_l1':
            self.loss_func = nn.SmoothL1Loss().cuda()
        else:
            self.loss_func = nn.MSELoss().cuda()
    
    def _reset_parameters(self, model):
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def compute_turn_weight(self,trajectory):
        """
        计算轨迹的转弯弧度损失权重
        Args:
            trajectory: Tensor of shape (batch_size, waypoints, 2)
        Returns:
            loss_weight: Tensor of shape (batch_size,)
        """
        # 提取前、中、后点（忽略首尾两点）
        prev_points = trajectory[:, :-2, :]  # (B, W-2, 2)
        curr_points = trajectory[:, 1:-1, :]  # (B, W-2, 2)
        next_points = trajectory[:, 2:, :]  # (B, W-2, 2)
        
        # 计算前向和后向向量
        v1 = curr_points - prev_points  # 向量从prev到curr
        v2 = next_points - curr_points  # 向量从curr到next
        
        # 计算向量点积和模长
        dot_product = torch.sum(v1 * v2, dim=2)  # (B, W-2)
        norm_v1 = torch.norm(v1, dim=2)  # (B, W-2)
        norm_v2 = torch.norm(v2, dim=2)  # (B, W-2)
        
        # 计算夹角余弦（防止除以零）
        cos_theta = dot_product / (norm_v1 * norm_v2 + 1e-6)
        theta = torch.acos(cos_theta)  # 弧度制夹角 (B, W-2)
        
        # 计算补角（π - θ），补角越大转弯越急
        alpha = torch.pi - theta
        
        # 取每个样本的补角均值作为权重
        loss_weight = torch.mean(alpha, dim=1)  # (B,)
        
        return loss_weight

    def determine_turn_direction(self, trajectory, threshold_ratio=0.07):
        if len(trajectory) < 3:
            return -1  # 表示无法判断
        
        sum_cross = 0.0
        total_length = 0.0
        
        for i in range(len(trajectory) - 2):
            p0 = trajectory[i]
            p1 = trajectory[i+1]
            p2 = trajectory[i+2]
            
            v1x, v1y = p1[0]-p0[0], p1[1]-p0[1]
            v2x, v2y = p2[0]-p1[0], p2[1]-p1[1]
            segment_length = max(abs(v1x)+abs(v1y), 1e-5)
            
            sum_cross += (v1x * v2y - v1y * v2x) / segment_length
            total_length += segment_length

        dynamic_threshold = total_length * threshold_ratio
        
        if abs(sum_cross) < dynamic_threshold:
            return 1  # 直行
        elif sum_cross > 0:
            return 0  # 左转
        else:
            return 2  # 右转
    def forward(self, data_dict):

        trajectory = data_dict['traj_ins']
        # angle = data_dict['angle']
        deltas = get_delta(trajectory)
        # action_stats = get_data_stats(deltas)
        # print_color(f"action_stats: {action_stats}", 'red')
        n_action = normalize_data(deltas, ACTION_STATS).to(self.device)
        
        bsz = trajectory.shape[0]
        noise = torch.randn(trajectory.shape, device=self.device)
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.num_train_timesteps, 
            (bsz,), device=self.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            n_action, noise, timesteps)
        src = data_dict['waypoints_feature']
        bs = src.shape[0]
        src = src.permute(0, 2, 1) # [bs, desc_dim, num_points]
        src = self.feature_encoder(src)
        print_color(f"src shape: {src.shape}", 'red')
        src = src.permute(2, 0, 1) # [num_points, bs, d_model]
        
        pos_embed = None
        if self.use_position_encoder:
            pos = data_dict['pos_feature']
            pos = pos.permute(0, 2, 1) # [bs, desc_dim, num_points]
            pos_embed = self.pos_encoder(pos)
            pos_embed = pos_embed.permute(2, 0, 1) # [num_points, bs, d_model]

        # obs_cond = src + pos_embed
        obs_cond = src
        obs_cond = einops.rearrange(obs_cond, 't b h -> b h t')
        print(f"obs_cond shape: {obs_cond.shape}")
        # obs_cond = einops.reduce(obs_cond, 'b h t -> b h', 'mean')
        obs_cond = obs_cond.reshape(bsz, -1)
        
        print(f"obs_cond shape: {obs_cond.shape}")

        # Predict the noise residual
        noise_pred = self.noise_pred_net(sample = noisy_trajectory, timestep = timesteps, global_cond = obs_cond) 

        # if self.use_transformer_encoder:
        #     src = self.transformer_encoder(src, pos=pos_embed)
        
        # if self.use_transformer_decoder:
        #     query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        #     tgt = torch.zeros_like(query_embed)
        #     hs = self.transformer_decoder(tgt, src, pos=pos_embed, query_pos=query_embed)
        #     hs = hs.permute(1, 2, 0) # [num_points, bs, d_model] -> [bs, d_model, num_points]
        # else:
        #     hs = src.permute(1, 2, 0) # [num_points, bs, d_model] -> [bs, d_model, num_points]

        # traj_pred = self.proj(hs)
        # data_dict['waypoints_pred'] = traj_pred.permute(0, 2, 1) # [bs, num_points, 2]


        # print_color(f"diffusion_output: {diffusion_output.shape}", 'red')
        loss = 0
        if self.training:
            # self.pred = data_dict['waypoints_pred']
            # self.target = data_dict['traj_ins_pixel_norm']
            pred = noise_pred
            self.target = noise
            loss = self.loss_func(pred, self.target)
            
            pred_traj = noisy_trajectory -  noise_pred 
            gt_traj = noisy_trajectory - noise
            
                    # 计算 vla_tag
            pred_vla_tags = []
            target_vla_tags = []
            for i in range(bsz):
                pred_vla_tag = self.determine_turn_direction(pred_traj[i].cpu().detach().numpy())
                target_vla_tag = self.determine_turn_direction(gt_traj[i].cpu().detach().numpy())
                pred_vla_tags.append(pred_vla_tag)
                target_vla_tags.append(target_vla_tag)
            # loss = (loss * angle).mean()
            # print_color(f"angle: {angle}", 'yellow')
                    # 将 vla_tag 转换为数值形式
            pred_vla_tags = torch.tensor(pred_vla_tags, dtype=torch.long, device=self.device)
            target_vla_tags = torch.tensor(target_vla_tags, dtype=torch.long, device=self.device)
            
                    # 添加一个全连接层来预测 vla_tag 的概率
            self.vla_tag_classifier = nn.Linear(obs_cond.shape[1], 3).to(self.device)
            
            # 使用 obs_cond 作为输入来预测 vla_tag 的概率
            vla_tag_logits = self.vla_tag_classifier(obs_cond)
            
            # 计算 vla_tag 的损失
            vla_tag_loss = F.cross_entropy(vla_tag_logits, target_vla_tags)
            
            # 将 vla_tag 损失添加到总损失中
            # loss += vla_tag_loss
            
        # else:
        # initialize action from Gaussian noise
        noisy_diffusion_output = torch.randn(trajectory.shape[0]*self.num_samples,
                                             trajectory.shape[1],
                                             trajectory.shape[2], 
                                             device=self.device) # [32,8,2]
        diffusion_output = noisy_diffusion_output
        # print_color(f"obs_cond: {obs_cond.shape}", 'green')
        obs_cond = obs_cond.repeat(self.num_samples, 1)
        # print_color(f"obs_cond: {obs_cond.shape}", 'red')
        # print_color(f"diffusion_output: {diffusion_output.shape}", 'red')
        
        # print_color(f"noise_scheduler.timesteps: {self.noise_scheduler.timesteps}", 'red')
        intermediate_samples = []
        self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
        for t in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred = self.noise_pred_net(sample = diffusion_output, timestep = t.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(self.device), global_cond = obs_cond)
            # inverse diffusion step (remove noise)
            diffusion_output = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=diffusion_output
            ).prev_sample # [32,8,2]
            intermediate_samples.append(unnormalize_data(diffusion_output, ACTION_STATS).to(self.device)) 


          
        data_dict['waypoints_pred'] = unnormalize_data(diffusion_output, ACTION_STATS).to(self.device) # [bs, num_points, 2]
        print_color(f"diffusion_output: {diffusion_output.shape}", 'red')
        data_dict['intermediate_samples'] = intermediate_samples
        
        
        # # 初始化最小损失和对应的轨迹索引
        min_loss_last = 0

        return data_dict,loss, min_loss_last

def MLP(channels: list, do_bn=True, drop_out=False):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
            if drop_out:
                layers.append(nn.Dropout(0.5))

    return nn.Sequential(*layers)

class Encoder(nn.Module):
    """ Encoding waypoint features using MLPs"""
    def __init__(self, input_dim, feature_dim, layers=[256, 128]):
        super(Encoder, self).__init__()
        self.encoder = MLP([input_dim] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, pts):
        return self.encoder(pts)

class TransformerEnocder(nn.Module):

    def __init__(self, encoder_layer, cfg):
        super().__init__()
        self.layers = _get_clones(encoder_layer, cfg.num_layers)
        self.num_layers = cfg.num_layers

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        return output

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, cfg):
        super().__init__()
        self.layers = _get_clones(decoder_layer, cfg.num_layers)
        self.num_layers = cfg.num_layers
        self.norm = None
        self.return_intermediate = cfg.get('return_intermediate', False)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output

class TransformerEncoderLayer(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        d_model = cfg.d_model
        dim_feedforward = cfg.dim_feedforward
        nhead = cfg.nhead
        dropout = cfg.dropout
        activation = cfg.activation
        normalize_before = cfg.get('normalize_before', False)

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerDecoderLayer(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        d_model = cfg.d_model
        nhead = cfg.nhead
        dropout = cfg.dropout
        activation = cfg.activation
        normalize_before = cfg.get('normalize_before', False)
        dim_feedforward = cfg.dim_feedforward

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
