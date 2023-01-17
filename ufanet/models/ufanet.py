import torch
import torch.nn as nn
from torchvision.transforms import Resize
from .common import ATBlock, CNNBlock, FeatureAggregation
from torchsummary import summary


class UFANet(nn.Module):

    def __init__(self, in_channels=1, num_classes=2, base_filter_dim=64, depth=4, attn=True, fam=True):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.base_filter_dim = base_filter_dim
        self.depth = depth
        self.attn = attn
        self.fam = fam
        self.mp = nn.MaxPool2d(2)
        self.down_blocks = nn.ModuleList(self.fill_in_down_blocks())
        self.up = nn.Upsample(scale_factor=2)
        if self.fam:
            self.fams = nn.ModuleList(self.fill_in_fam_down_blocks())
        self.bottle_neck = ATBlock(self.base_filter_dim * 2**(depth-1), self.base_filter_dim * 2**depth) if attn else \
            CNNBlock(self.base_filter_dim * 2**(depth-1), self.base_filter_dim * 2**depth)
        self.up_blocks = nn.ModuleList(self.fill_in_up_blocks())
        self.up_sample_blocks = nn.ModuleList(self.fill_in_upsample_blocks())
        self.fin_cnn = nn.Conv2d(self.base_filter_dim, self.num_classes, kernel_size=1)

    def fill_in_down_blocks(self):
        blocks = []
        for level in range(self.depth):
            if level == 0:
                block = ATBlock(self.in_channels, self.base_filter_dim) if self.attn \
                    else CNNBlock(self.in_channels, self.base_filter_dim)
            else:
                block = ATBlock(self.base_filter_dim * 2**(level-1), self.base_filter_dim * 2**level) if self.attn \
                    else CNNBlock(self.base_filter_dim * 2**(level-1), self.base_filter_dim * 2**level)
            blocks.append(block)
        return blocks

    def fill_in_fam_down_blocks(self):
        blocks = []
        for level in range(self.depth - 1):
            block = FeatureAggregation(self.base_filter_dim * 2**level, self.base_filter_dim * 2**(level+1))
            blocks.append(block)
        return blocks

    def fill_in_up_blocks(self):
        blocks = []
        for level in range(self.depth):
            block = ATBlock(self.base_filter_dim * 2**(level+1), self.base_filter_dim * 2**level) if self.attn \
                else CNNBlock(self.base_filter_dim * 2**(level+1), self.base_filter_dim * 2**level)
            blocks.append(block)
        return blocks

    def fill_in_upsample_blocks(self):
        blocks = []
        for level in range(self.depth):
            block = nn.Conv2d(self.base_filter_dim * 2**(level+1),
                              self.base_filter_dim * 2**level,
                              kernel_size=3, padding=1, bias=False)
            blocks.append(block)
        return blocks

    @staticmethod
    def skip_connection(block1, block2):
        curr_shape = block2.shape[-1]
        block1 = Resize((curr_shape, curr_shape))(block1)
        concatenated_block = torch.cat([block1, block2], dim=1)
        return concatenated_block

    def up_sample(self, batch):
        # curr_shape = batch.shape[-1]
        # return Resize((curr_shape*2, curr_shape*2))(batch)
        return self.up(batch)

    def prepare_pre_final_block(self, image_batch):
        upd_batches = []
        if self.fam:
            fam_batches = []
        for index, down_block in enumerate(self.down_blocks):
            image_batch = down_block(image_batch)
            if self.fam:
                fam_batches.append(image_batch)
                if index == 0:
                    image_batch = self.mp(image_batch)
                else:
                    image_batch = self.fams[index-1](fam_batches[len(fam_batches) - 2], image_batch)
            else:
                image_batch = self.mp(image_batch)
            upd_batches.append(image_batch)
        image_batch = self.bottle_neck(image_batch)
        for up_block, upsample_block in zip(reversed(self.up_blocks),
                                            reversed(self.up_sample_blocks)):
            image_batch = self.up_sample(image_batch)
            image_batch = upsample_block(image_batch)
            image_batch = self.skip_connection(upd_batches[-1], image_batch)
            upd_batches.pop()
            image_batch = up_block(image_batch)

        return image_batch

    def forward(self, image_batch):
        image_batch = self.prepare_pre_final_block(image_batch=image_batch)
        return self.fin_cnn(image_batch)


def test():
    model = UFANet(attn=False, fam=True, depth=3)
    rand_data = torch.rand(5, 1, 300, 300)
    print(model(rand_data).shape)
    summary(model, rand_data.shape[1:], device="cpu")


if __name__ == "__main__":
    test()