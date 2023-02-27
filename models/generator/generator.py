import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.misc import weights_init
from models.generator.cfa import CFA
from models.generator.bigff import BiGFF
from models.generator.pconv import PConvBNActiv
from models.generator.WIRW import WIRW
from models.generator.projection import Feature2Structure, Feature2Texture
from models.generator.model.fuse import FusionBlock_res

class Generator(nn.Module):

    def __init__(self, image_in_channels=3, edge_in_channels=2, out_channels=3, init_weights=True):
        super(Generator, self).__init__()

        self.freeze_ec_bn = False

        # -----------------------
        # texture encoder-decoder
        # -----------------------
        self.ec_texture_1 = PConvBNActiv(image_in_channels, 64, bn=False, sample='down-7')
        self.ec_texture_2 = PConvBNActiv(64, 128, sample='down-5')
        self.ec_texture_3 = PConvBNActiv(128, 256, sample='down-5')
        self.ec_texture_4 = PConvBNActiv(256, 512, sample='down-3')
        self.ec_texture_5 = PConvBNActiv(512, 512, sample='down-3')
        self.ec_texture_6 = PConvBNActiv(512, 512, sample='down-3')
        self.ec_texture_7 = PConvBNActiv(512, 512, sample='down-3')

        self.dc_texture_7 = PConvBNActiv(512 + 512, 512, activ='leaky')
        self.dc_texture_6 = PConvBNActiv(512 + 512, 512, activ='leaky')
        self.dc_texture_5 = PConvBNActiv(512 + 512, 512, activ='leaky')
        self.dc_texture_4 = PConvBNActiv(512 + 256, 256, activ='leaky')

        self.dc_texture_3 = PConvBNActiv(256 + 128, 128, activ='leaky')
        self.dc_texture_2 = PConvBNActiv(128 + 64, 64, activ='leaky')
        self.dc_texture_1 = PConvBNActiv(64 + out_channels, 64, activ='leaky')

        # -------------------------
        # structure encoder-decoder
        # -------------------------
        self.ec_structure_1 = PConvBNActiv(edge_in_channels, 64, bn=False, sample='down-7')
        self.ec_structure_2 = PConvBNActiv(64, 128, sample='down-5')
        self.ec_structure_3 = PConvBNActiv(128, 256, sample='down-5')
        self.ec_structure_4 = PConvBNActiv(256, 512, sample='down-3')
        self.ec_structure_5 = PConvBNActiv(512, 512, sample='down-3')
        self.ec_structure_6 = PConvBNActiv(512, 512, sample='down-3')
        self.ec_structure_7 = PConvBNActiv(512, 512, sample='down-3')

        self.dc_structure_7 = PConvBNActiv(512 + 512, 512, activ='leaky')
        self.dc_structure_6 = PConvBNActiv(512 + 512, 512, activ='leaky')
        self.dc_structure_5 = PConvBNActiv(512 + 512, 512, activ='leaky')
        self.dc_structure_4 = PConvBNActiv(512 + 256, 256, activ='leaky')
        self.dc_structure_3 = PConvBNActiv(256 + 128, 128, activ='leaky')
        self.dc_structure_2 = PConvBNActiv(128 + 64, 64, activ='leaky')
        self.dc_structure_1 = PConvBNActiv(64 + 2, 64, activ='leaky')

        # -------------------
        #WIRW module
        # -------------------
        self.sdiss4=WIRW(n_feats=512)
        self.sdiss5=WIRW(n_feats=512)
        self.sdiss6=WIRW(n_feats=512)

        self.structure_feature_projection = Feature2Structure()
        self.texture_feature_projection = Feature2Texture()
        self.bigff = BiGFF(in_channels=64, out_channels=64)
        self.out_layer = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1),
            nn.Tanh()
        )

        # -------------------
        # SAT module
        # -------------------
        self.fuse=FusionBlock_res(64,256)
        if init_weights:
           self.apply(weights_init())

    def forward(self, input_image, input_edge, mask):

        ec_textures = {}
        ec_structures = {}

        input_texture_mask = torch.cat((mask, mask, mask), dim=1)
        ec_textures['ec_t_0'], ec_textures['ec_t_masks_0'] = input_image, input_texture_mask
        ec_textures['ec_t_1'], ec_textures['ec_t_masks_1'] = self.ec_texture_1(ec_textures['ec_t_0'], ec_textures['ec_t_masks_0'])
        ec_textures['ec_t_2'], ec_textures['ec_t_masks_2'] = self.ec_texture_2(ec_textures['ec_t_1'], ec_textures['ec_t_masks_1'])
        ec_textures['ec_t_3'], ec_textures['ec_t_masks_3'] = self.ec_texture_3(ec_textures['ec_t_2'], ec_textures['ec_t_masks_2'])
        ec_textures['ec_t_4'], ec_textures['ec_t_masks_4'] = self.ec_texture_4(ec_textures['ec_t_3'], ec_textures['ec_t_masks_3'])
        ec_textures['ec_t_5'], ec_textures['ec_t_masks_5'] = self.ec_texture_5(ec_textures['ec_t_4'], ec_textures['ec_t_masks_4'])
        ec_textures['ec_t_6'], ec_textures['ec_t_masks_6'] = self.ec_texture_6(ec_textures['ec_t_5'], ec_textures['ec_t_masks_5'])
        ec_textures['ec_t_7'], ec_textures['ec_t_masks_7'] = self.ec_texture_7(ec_textures['ec_t_6'], ec_textures['ec_t_masks_6'])

        input_structure_mask = torch.cat((mask, mask), dim=1)
        ec_structures['ec_s_0'], ec_structures['ec_s_masks_0'] = input_edge, input_structure_mask
        ec_structures['ec_s_1'], ec_structures['ec_s_masks_1'] = self.ec_structure_1(ec_structures['ec_s_0'], ec_structures['ec_s_masks_0'])
        ec_structures['ec_s_2'], ec_structures['ec_s_masks_2'] = self.ec_structure_2(ec_structures['ec_s_1'], ec_structures['ec_s_masks_1'])
        ec_structures['ec_s_3'], ec_structures['ec_s_masks_3'] = self.ec_structure_3(ec_structures['ec_s_2'], ec_structures['ec_s_masks_2'])
        ec_structures['ec_s_4'], ec_structures['ec_s_masks_4'] = self.ec_structure_4(ec_structures['ec_s_3'], ec_structures['ec_s_masks_3'])
        ec_structures['ec_s_4']=self.sdiss4(ec_structures['ec_s_4'])
        ec_structures['ec_s_5'], ec_structures['ec_s_masks_5'] = self.ec_structure_5(ec_structures['ec_s_4'], ec_structures['ec_s_masks_4'])
        ec_structures['ec_s_5'] = self.sdiss4(ec_structures['ec_s_5'])
        ec_structures['ec_s_6'], ec_structures['ec_s_masks_6'] = self.ec_structure_6(ec_structures['ec_s_5'], ec_structures['ec_s_masks_5'])
        ec_structures['ec_s_6'] = self.sdiss4(ec_structures['ec_s_6'])
        ec_structures['ec_s_7'], ec_structures['ec_s_masks_7'] = self.ec_structure_7(ec_structures['ec_s_6'], ec_structures['ec_s_masks_6'])

        dc_texture, dc_tecture_mask = ec_structures['ec_s_7'], ec_structures['ec_s_masks_7']
        for _ in range(7, 0, -1):
            ec_texture_skip = 'ec_t_{:d}'.format(_ - 1)
            ec_texture_masks_skip = 'ec_t_masks_{:d}'.format(_ - 1)
            dc_conv = 'dc_texture_{:d}'.format(_)
            dc_texture = F.interpolate(dc_texture, scale_factor=2, mode='bilinear')
            dc_tecture_mask = F.interpolate(dc_tecture_mask, scale_factor=2, mode='nearest')
            dc_texture = torch.cat((dc_texture, ec_textures[ec_texture_skip]), dim=1)
            dc_tecture_mask = torch.cat((dc_tecture_mask, ec_textures[ec_texture_masks_skip]), dim=1)
            dc_texture, dc_tecture_mask = getattr(self, dc_conv)(dc_texture, dc_tecture_mask)

        dc_structure, dc_structure_masks = ec_textures['ec_t_7'], ec_textures['ec_t_masks_7']
        for _ in range(7, 0, -1):
            ec_structure_skip = 'ec_s_{:d}'.format(_ - 1)
            ec_structure_masks_skip = 'ec_s_masks_{:d}'.format(_ - 1)
            dc_conv = 'dc_structure_{:d}'.format(_)
            dc_structure = F.interpolate(dc_structure, scale_factor=2, mode='bilinear')
            dc_structure_masks = F.interpolate(dc_structure_masks, scale_factor=2, mode='nearest')
            dc_structure = torch.cat((dc_structure, ec_structures[ec_structure_skip]), dim=1)
            dc_structure_masks = torch.cat((dc_structure_masks, ec_structures[ec_structure_masks_skip]), dim=1)
            dc_structure, dc_structure_masks = getattr(self, dc_conv)(dc_structure, dc_structure_masks)
        # -------------------
        # Projection Function
        # -------------------
        projected_image = self.texture_feature_projection(dc_texture) # 1 128 256 256
        projected_edge = self.structure_feature_projection(dc_structure) # 1 64 256 256
        text,struc = self.bigff(dc_texture, dc_structure) #  1 64 256 256
        output=self.fuse(text,struc)
        output = self.out_layer(output)
        return output, projected_image, projected_edge

    def train(self, mode=True):

        super().train(mode)

        if self.freeze_ec_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
