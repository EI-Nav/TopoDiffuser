## 融合diffusion policy中
关键文件： 
- diffusion_transformer_lowdim_policy.py
- train-diffusion_lowdim_workspace.py
# 前向加噪过程
对干净的trajectory添加噪声
```python
trajectory = action
action = nbatch['action']
batch = self.normalize(batch)

```

## 制作数据集已完成
数据集中需要的元素包含：
- utm_pose
- 真实轨迹
- osm route
1. 需要将数据存储到pkl文件中，
multi_generate_json.py文件
2. 然后读取pkl文件，以规定json格式输出。
read_from_pkl.ipynb文件
3. 雷达数据需要进行旋转
read_from_pkl.ipynb
# Trajectory Prediction for Autonomous Driving with Topometric Map

![image](https://github.com/Jiaolong/trajectory-prediction/tree/main/data/kitti/traj_pred_kitti10.gif)

Repository for the paper ["Trajectory Prediction for Autonomous Driving with Topometric Map"](https://arxiv.org/abs/2105.03869).
```
@inproceedings{traj-pred:2022,
  title={Trajectory Prediction for Autonomous Driving with Topometric Map},
  author={J. Xu, L. Xiao, D. Zhao etal},
  booktitle={ICRA},
  year={2022}
}
```

## Requirements

- python 3.6+

- pytorch 1.4+

## Install build requirements

```shell
pip install -v -e .  # or "python setup.py develop"
```
该行语句会建立软链接

## Pretrained models

Pre-trained weights can be downloaded [here](https://pan.baidu.com/s/1Ns7qjW352rMXJhleGJN2TQ)(code: uf9g)

## Dataset

```
├── datasets
│   └── KITTI_RAW
        └── trajectory_prediction
            ├── 07
            └── 10
```

Testing dataset kitti-10 can be downloaded [here](https://pan.baidu.com/s/1DrPRNWfMOy7JMc_TOzdV7w)(code: kbuf)

## Train & Test

### Train

* Train with multiple GPUs:
```shell script
sh tools/scripts/dist_train.sh ${NUM_GPUS} -c ${CONFIG_FILE}
```

* Train with a single GPU:
```shell script
python tools/train.py --cfg config/trajectory_prediction/transformer.yaml
```

### Test

* Test with a pretrained model:
```shell script
python tools/test.py --cfg config/trajectory_prediction/transformer.yaml --ckpt cache/transformer_epoch_120.pth
```
