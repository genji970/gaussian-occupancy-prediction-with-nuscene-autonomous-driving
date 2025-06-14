from down_conference.config import config

# 1. backbone 설정
backbone_cfg = dict(
    type='SwinTransformer',
    embed_dims=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=7,
    mlp_ratio=4.,
    qkv_bias=True,
    drop_path_rate=0.2,
    patch_norm=True,
    out_indices=(0, 1, 2, 3),
    with_cp=False,
    init_cfg=dict(type='Pretrained', checkpoint='https://download.openmmlab.com/.../swin_tiny.pth')
)