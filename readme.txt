2021.8.20 v1 DenseNet Pytorch实现；

2021.9.13 v2.0 在v1基础上增加APEX DDP并行加速,把模型中的标准卷积换成深度可分离卷积；

2021.9.30 v2.1 v2.0无法使用混合精度，会报溢出问题，v2.1注册soft为单精度，使用amp.state_dict为模型文件，使用凯明权重初始化，增加SWISH和PReLU激活函数
