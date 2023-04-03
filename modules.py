from torch import nn


class CIFARResNet(nn.Module):
    def __init__(self, num_blocks=(2, 2, 2), inplanes=16) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(3, inplanes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.act1 = nn.ReLU(inplace=True)

        stages = []
        for stage_id, num_block in enumerate(num_blocks):
            blocks = []
            stride = 1 if stage_id == 0 else 2
            out_channels = (stage_id + 2) * inplanes
            for block_id in range(num_block):
                stride = stride if block_id == 0 else 1
                if block_id == 0:
                    block = nn.Sequential(
                        nn.Conv2d(inplanes, out_channels, kernel_size=3,
                                  stride=stride, padding=1))
                    blocks.append()
