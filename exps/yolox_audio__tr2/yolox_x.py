# encoding: utf-8
import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist

from yolox.exp import Exp as MyExp
from yolox.data import get_yolox_datadir

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 3
        self.depth = 1.33
        self.width = 1.25
# ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        self.data_num_workers = 4
        self.input_size = (256, 512)  # (height, width)
        # Actual multiscale ranges: [640-5*32, 640+5*32].
        # To disable multiscale training, set the
        # self.multiscale_range to 0.
        self.multiscale_range = 0
        self.train_path = '/data/AIGC_3rd_2021/GIST_tr2_veryhard5000_all_tr2'
        self.val_path = '/data/AIGC_3rd_2021/GIST_tr2_veryhard500'
        self.train_ann = "label_coco_bbox.json"
        self.val_ann = "label_coco_bbox.json"

        # --------------- transform config ----------------- #
        self.mosaic_prob = 0.0
        self.mixup_prob = 0.2
        self.hsv_prob = 0.0
        self.flip_prob = 0.0
        self.degrees = 0.0
        self.translate = 0.0
        self.mosaic_scale = (0.5, 1.5)
        self.mixup_scale = (0.5, 1.5)
        self.shear = 0.0
        self.perspective = 0.0
        self.enable_mixup = True

        # --------------  training config --------------------- #
        self.warmup_epochs = 1
        self.max_epoch = 200
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05
        self.ema = True

        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 10
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        self.test_size = (256, 512)
        self.test_conf = 0.01
        self.nmsthre = 0.65

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data.datasets.intflow import INTFLOWDataset
        from yolox.data.datasets.mosaicdetection import MosaicDetection
        from yolox.data.data_augment import TrainTransform
        from yolox.data.dataloading import DataLoader
        from yolox.data.samplers import InfiniteSampler, YoloBatchSampler
        import torch.distributed as dist

        dataset = INTFLOWDataset(
                data_dir=self.train_path,
                json_file=self.train_ann,
                name="img",
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=10,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob),
                rotation=False,
                compatible_coco=True,
                cache=cache_img,
        )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=20,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import INTFLOWDataset, ValTransform

        valdataset = INTFLOWDataset(
            data_dir=self.val_path,
            json_file=self.val_ann,
            name="img",
            img_size=self.input_size,
            preproc=ValTransform(),
            compatible_coco=True,
            rotation=False,
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from yolox.evaluators import INTFLOWEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev)
        evaluator = INTFLOWEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator

    def eval(self, model, evaluator, is_distributed, half=False):
        return evaluator.evaluate(model, is_distributed, half)
