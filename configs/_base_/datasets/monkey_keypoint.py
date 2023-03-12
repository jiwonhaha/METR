dataset_type = 'opera.NHPDataset'
data_root = 'monkey_dataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', to_float32=True),
    dict(
        type='opera.LoadAnnotations',
        with_bbox=True,
        with_keypoint=True,
        with_area=True),
    dict(
        type='mmdet.PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='opera.KeypointRandomAffine',
        max_rotate_degree=30.0,
        max_translate_ratio=0.0,
        scaling_ratio_range=(1.0, 1.0),
        max_shear_degree=0.0,
        border_val=[103.53, 116.28, 123.675]),
    dict(type='opera.RandomFlip', flip_ratio=0.5),
    dict(
        type='mmdet.AutoAugment',
        policies=[[{
            'type': 'opera.Resize',
            'img_scale': [(400, 1400), (1400, 1400)],
            'multiscale_mode': 'range',
            'keep_ratio': True
        }],
                  [{
                      'type': 'opera.Resize',
                      'img_scale': [(400, 4200), (500, 4200), (600, 4200)],
                      'multiscale_mode': 'value',
                      'keep_ratio': True
                  }, {
                      'type': 'opera.RandomCrop',
                      'crop_type': 'absolute_range',
                      'crop_size': (384, 600),
                      'allow_negative_crop': True
                  }, {
                      'type': 'opera.Resize',
                      'img_scale': [(400, 1400), (1400, 1400)],
                      'multiscale_mode': 'range',
                      'override': True,
                      'keep_ratio': True
                  }]]),
    dict(
        type='mmdet.Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='mmdet.Pad', size_divisor=1),
    dict(
        type='opera.DefaultFormatBundle',
        extra_keys=['gt_keypoints', 'gt_areas']),
    dict(
        type='mmdet.Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_keypoints', 'gt_areas'])
]
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(
        type='mmdet.MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='mmdet.Resize', keep_ratio=True),
            dict(type='mmdet.RandomFlip'),
            dict(
                type='mmdet.Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='mmdet.Pad', size_divisor=1),
            dict(type='mmdet.ImageToTensor', keys=['img']),
            dict(type='mmdet.Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='NHPDataset',
        ann_file='monkey_dataset/cocoMonkeyTrain.json',
        img_prefix='monkey_dataset/train',
        pipeline=[
            dict(type='mmdet.LoadImageFromFile', to_float32=True),
            dict(
                type='opera.LoadAnnotations',
                with_bbox=True,
                with_keypoint=True,
                with_area=True),
            dict(
                type='mmdet.PhotoMetricDistortion',
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18),
            dict(
                type='opera.KeypointRandomAffine',
                max_rotate_degree=30.0,
                max_translate_ratio=0.0,
                scaling_ratio_range=(1.0, 1.0),
                max_shear_degree=0.0,
                border_val=[103.53, 116.28, 123.675]),
            dict(type='opera.RandomFlip', flip_ratio=0.5),
            dict(
                type='mmdet.AutoAugment',
                policies=[[{
                    'type': 'opera.Resize',
                    'img_scale': [(400, 1400), (1400, 1400)],
                    'multiscale_mode': 'range',
                    'keep_ratio': True
                }],
                          [{
                              'type': 'opera.Resize',
                              'img_scale': [(400, 4200), (500, 4200),
                                            (600, 4200)],
                              'multiscale_mode': 'value',
                              'keep_ratio': True
                          }, {
                              'type': 'opera.RandomCrop',
                              'crop_type': 'absolute_range',
                              'crop_size': (384, 600),
                              'allow_negative_crop': True
                          }, {
                              'type': 'opera.Resize',
                              'img_scale': [(400, 1400), (1400, 1400)],
                              'multiscale_mode': 'range',
                              'override': True,
                              'keep_ratio': True
                          }]]),
            dict(
                type='mmdet.Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='mmdet.Pad', size_divisor=1),
            dict(
                type='opera.DefaultFormatBundle',
                extra_keys=['gt_keypoints', 'gt_areas']),
            dict(
                type='mmdet.Collect',
                keys=[
                    'img', 'gt_bboxes', 'gt_labels', 'gt_keypoints', 'gt_areas'
                ])
        ],
        classes=('monkey', )),
    val=dict(
        type='NHPDataset',
        ann_file='monkey_dataset/cocoMonkeyVal.json',
        img_prefix='monkey_dataset/val',
        pipeline=[
            dict(type='mmdet.LoadImageFromFile'),
            dict(
                type='mmdet.MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='mmdet.Resize', keep_ratio=True),
                    dict(type='mmdet.RandomFlip'),
                    dict(
                        type='mmdet.Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='mmdet.Pad', size_divisor=1),
                    dict(type='mmdet.ImageToTensor', keys=['img']),
                    dict(type='mmdet.Collect', keys=['img'])
                ])
        ],
        classes=('monkey', )),
    test=dict(
        type='NHPDataset',
        ann_file='monkey_dataset/cocoMonkeyVal.json',
        img_prefix='monkey_dataset/val',
        pipeline=[
            dict(type='mmdet.LoadImageFromFile'),
            dict(
                type='mmdet.MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='mmdet.Resize', keep_ratio=True),
                    dict(type='mmdet.RandomFlip'),
                    dict(
                        type='mmdet.Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='mmdet.Pad', size_divisor=1),
                    dict(type='mmdet.ImageToTensor', keys=['img']),
                    dict(type='mmdet.Collect', keys=['img'])
                ])
        ],
        classes=('monkey', )))
evaluation = dict(interval=1, metric='keypoints')
classes = ('monkey', )
model = dict(
    type='opera.PETR',
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='mmdet.ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    bbox_head=dict(
        type='opera.PETRHead',
        num_query=300,
        num_classes=1,
        in_channels=2048,
        sync_cls_avg_factor=True,
        with_kpt_refine=True,
        as_two_stage=True,
        transformer=dict(
            type='opera.PETRTransformer',
            encoder=dict(
                type='mmcv.DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='mmcv.BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='mmcv.MultiScaleDeformableAttention',
                        embed_dims=256),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='opera.PetrTransformerDecoder',
                num_layers=3,
                return_intermediate=True,
                transformerlayers=dict(
                    type='mmcv.DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='mmcv.MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='opera.MultiScaleDeformablePoseAttention',
                            embed_dims=256)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            hm_encoder=dict(
                type='mmcv.DetrTransformerEncoder',
                num_layers=1,
                transformerlayers=dict(
                    type='mmcv.BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='mmcv.MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_levels=1),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            refine_decoder=dict(
                type='mmcv.DeformableDetrTransformerDecoder',
                num_layers=2,
                return_intermediate=True,
                transformerlayers=dict(
                    type='mmcv.DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='mmcv.MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='mmcv.MultiScaleDeformableAttention',
                            embed_dims=256,
                            im2col_step=128)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='mmcv.SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_kpt=dict(type='mmdet.L1Loss', loss_weight=70.0),
        loss_kpt_rpn=dict(type='mmdet.L1Loss', loss_weight=70.0),
        loss_oks=dict(type='opera.OKSLoss', loss_weight=2.0),
        loss_hm=dict(type='opera.CenterFocalLoss', loss_weight=4.0),
        loss_kpt_refine=dict(type='mmdet.L1Loss', loss_weight=80.0),
        loss_oks_refine=dict(type='opera.OKSLoss', loss_weight=3.0)),
    train_cfg=dict(
        assigner=dict(
            type='opera.PoseHungarianAssigner',
            cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
            kpt_cost=dict(type='opera.KptL1Cost', weight=70.0),
            oks_cost=dict(type='opera.OksCost', weight=7.0))),
    test_cfg=dict(max_per_img=100))
optimizer = dict(
    type='AdamW',
    lr=0.0002,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.1),
            sampling_offsets=dict(lr_mult=0.1),
            reference_points=dict(lr_mult=0.1))))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(policy='step', step=[80])
runner = dict(type='EpochBasedRunner', max_epochs=100)
work_dir = 'monkeyDir'
auto_resume = True
gpu_ids = [0]
