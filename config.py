import torch

config = {

    "device" : 'cuda' if torch.cuda.is_available() else 'cpu',

    "lifter_cfg_type": 'GaussianLifter3D' ,
    "lifter_in_channels" : 256 ,
    "lifter_num_levels" :4 ,
    "lifter_num_gaussians" : 128 ,
    "lifter_hidden_dim" : 128 ,

    "flag" : "CAMERA" ,
    "backbone_batch_num" : 8,
    "neck_batch_num" : 8,

    "lifter_num_anchor" : 25600,
    "lifter_feat_dim" : 256,
    "lifter_hidden_dim" : 128,

    "save_root" : "D:"

}