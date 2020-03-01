#FPN_RESNET   、  FPN_RESNET-SE  、  FPN_RESNET-CBAM

##fpn的结构如下

![图片名称可省略.jpg](https://github.com/youwh-PIRI/fpn_resnet-resnet-se_resnet-cbam/blob/master/fpn.JPG)

##resnet的结构如下

![图片名称可省略.jpg](https://github.com/youwh-PIRI/fpn_resnet-resnet-se_resnet-cbam/blob/master/resnet.JPG)

##resnet-se的结构如下

![图片名称可省略.jpg](https://github.com/youwh-PIRI/fpn_resnet-resnet-se_resnet-cbam/blob/master/resnet-se.JPG)

##resnet-cbam的结构如下

![图片名称可省略.jpg](https://github.com/youwh-PIRI/fpn_resnet-resnet-se_resnet-cbam/blob/master/resnet-cbam-1.JPG)

![图片名称可省略.jpg](https://github.com/youwh-PIRI/fpn_resnet-resnet-se_resnet-cbam/blob/master/resnet-cbam-2.JPG)



下面是FPN_RESNET-CBAM模型的结构模块以及每层的输出和参数量大小，以此为例
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu1): ReLU(inplace)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): Bottleneck(
      (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      (downsample): Sequential(
        (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (cbam): CBAM(
        (ChannelGate): ChannelGate(
          (mlp): Sequential(
            (0): Flatten()
            (1): Linear(in_features=256, out_features=16, bias=True)
            (2): ReLU()
            (3): Linear(in_features=16, out_features=256, bias=True)
          )
        )
        (SpatialGate): SpatialGate(
          (compress): ChannelPool()
          (spatial): BasicConv(
            (conv): Conv2d(2, 1, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
            (bn): BatchNorm2d(1, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          )
        )
      )
    )
    (1): Bottleneck(
      (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      (cbam): CBAM(
        (ChannelGate): ChannelGate(
          (mlp): Sequential(
            (0): Flatten()
            (1): Linear(in_features=256, out_features=16, bias=True)
            (2): ReLU()
            (3): Linear(in_features=16, out_features=256, bias=True)
          )
        )
        (SpatialGate): SpatialGate(
          (compress): ChannelPool()
          (spatial): BasicConv(
            (conv): Conv2d(2, 1, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
            (bn): BatchNorm2d(1, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          )
        )
      )
    )
    (2): Bottleneck(
      (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace)
      (cbam): CBAM(
        (ChannelGate): ChannelGate(
          (mlp): Sequential(
            (0): Flatten()
            (1): Linear(in_features=256, out_features=16, bias=True)
            (2): ReLU()
            (3): Linear(in_features=16, out_features=256, bias=True)
          )
        )
        (SpatialGate): SpatialGate(
          (compress): ChannelPool()
          (spatial): BasicConv(
            (conv): Conv2d(2, 1, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
            (bn): BatchNorm2d(1, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
          )
        )
      )
    )
  )
Total params: 33,691,367
Trainable params: 33,691,367
Non-trainable params: 0

Input size (MB): 0.57
Forward/backward pass size (MB): 480.95
Params size (MB): 128.52
Estimated Total Size (MB): 610.04
# fpn_resnet_resnet-se_resnet-cbam
