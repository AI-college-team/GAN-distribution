import torch
import torch.nn as nn

def conv_block1_1(in_dim,out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim, kernel_size=3, stride=1, padding=1)
    )
    return model

def conv_block1_2(in_dim,out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm2d(out_dim),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return model

def conv_trans_block(in_dim,out_dim):
    model = nn.Sequential(
        nn.ConvTranspose2d(in_dim,out_dim, kernel_size=3, stride=2, padding=1,output_padding=1),
        nn.InstanceNorm2d(out_dim),
        nn.LeakyReLU(0.2, inplace=True),
    )
    return model

def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool

def conv_block_2(in_dim,out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm2d(out_dim),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(out_dim,out_dim, kernel_size=3, stride=1, padding=1),
        nn.InstanceNorm2d(out_dim),
        nn.LeakyReLU(0.2, inplace=True),
    )
    return model

class Detector(nn.Module):
    def __init__(self, in_dim=3, out_dim=1, num_filter=64):
        super(Detector, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filter = num_filter

        print("\n------Initiating Detector------\n")

        self.down_1 = conv_block1_1(self.in_dim, self.num_filter)
        self.pool_1 = maxpool()
        self.down_1 = conv_block1_2(self.in_dim, self.num_filter)
        self.pool_1 = maxpool()
        self.down_2 = conv_block_2(self.num_filter * 1, self.num_filter * 2)
        self.pool_2 = maxpool()
        self.down_3 = conv_block_2(self.num_filter * 2, self.num_filter * 4)
        self.pool_3 = maxpool()
        self.down_4 = conv_block_2(self.num_filter * 4, self.num_filter * 8)
        self.pool_4 = maxpool()

        self.bridge = conv_block_2(self.num_filter * 8, self.num_filter * 16)

        self.trans_1 = conv_trans_block(self.num_filter * 16, self.num_filter * 8)
        self.up_1 = conv_block_2(self.num_filter * 16, self.num_filter * 8)
        self.trans_2 = conv_trans_block(self.num_filter * 8, self.num_filter * 4)
        self.up_2 = conv_block_2(self.num_filter * 8, self.num_filter * 4)
        self.trans_3 = conv_trans_block(self.num_filter * 4, self.num_filter * 2)
        self.up_3 = conv_block_2(self.num_filter * 4, self.num_filter * 2)
        self.trans_4 = conv_trans_block(self.num_filter * 2, self.num_filter * 1)
        self.up_4 = conv_block_2(self.num_filter * 2, self.num_filter * 1)

        self.out = nn.Sequential(
            nn.Conv2d(self.num_filter, self.out_dim, 3, 1, 1),
            nn.Sigmoid()  # 필수는 아님
        )

    def forward(self, input):
        down_1 = self.down_1(input)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)
        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)

        bridge = self.bridge(pool_4)

        trans_1 = self.trans_1(bridge)
        concat_1 = torch.cat([trans_1, down_4], dim=1)
        up_1 = self.up_1(concat_1)
        trans_2 = self.trans_2(up_1)
        concat_2 = torch.cat([trans_2, down_3], dim=1)
        up_2 = self.up_2(concat_2)
        trans_3 = self.trans_3(up_2)
        concat_3 = torch.cat([trans_3, down_2], dim=1)
        up_3 = self.up_3(concat_3)
        trans_4 = self.trans_4(up_3)
        concat_4 = torch.cat([trans_4, down_1], dim=1)
        up_4 = self.up_4(concat_4)
        out = self.out(up_4)
        return out
