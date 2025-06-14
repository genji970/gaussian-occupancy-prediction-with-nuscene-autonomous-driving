from down_conference.config import config

neck_cfg = dict(
    type='FPN',
    in_channels=[768, 384, 192, 96], # reverse
    out_channels=256,
    num_outs=4
)