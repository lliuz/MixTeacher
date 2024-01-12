_base_ = "base.py"

classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        sup=dict(
            type="CocoDataset",
            ann_file="data/voc/annotations_json/voc07_trainval.json",
            img_prefix="data/voc/",
            classes=classes,
        ),
        unsup=dict(
            _delete_=True,
            type="ConcatDataset",
            datasets=[
                dict(
                    type="CocoDataset",
                    ann_file="data/voc/annotations_json/voc12_trainval.json",
                    img_prefix="data/voc/",
                    classes=classes,
                    pipeline="${unsup_pipeline}",
                    filter_empty_gt=False,
                ),
                dict(
                    type="CocoDataset",
                    ann_file="data/coco/annotations/semi_supervised/instances_unlabeledtrainval20class.json",
                    img_prefix="data/coco/",
                    classes=classes,
                    pipeline="${unsup_pipeline}",
                    filter_empty_gt=False,
                )
            ]
        ),
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 1],
        )
    ),
    val=dict(
        ann_file="data/voc/annotations_json/voc07_test.json",
        img_prefix="data/voc/",
        classes=classes,
    ),
    test=dict(
        ann_file="data/voc/annotations_json/voc07_test.json",
        img_prefix="data/voc/",
        classes=classes,
    ),
)

lr_config = dict(step=[40000])
runner = dict(max_iters=40000)

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=20,
        )
    )
)
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="pre_release",
                name="${cfg_name}",
                config=dict(
                    work_dirs="${work_dir}",
                    total_step="${runner.max_iters}",
                ),
            ),
            by_epoch=False,
        ),
    ],
)
