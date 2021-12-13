#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import datetime
import os
import time
from loguru import logger

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import numpy as np
import torch_pruning as tp
from yolox_audio.models import *
from yolox_audio.data import DataPrefetcher
from yolox_audio.utils import (
    MeterBuffer,
    ModelEMA,
    all_reduce_norm,
    get_local_rank,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    is_parallel,
    load_ckpt,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize
)


class Pruner:
    def __init__(self, exp, args):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        self.args = args

        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank)

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_ap = 0

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

    def pruning(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)
        
        if self.args.model == None:
            model = self.exp.get_model()
        else:
            model = self.exp.get_model_pruning(self.args.model)
        model = self.resume_train(model)

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)
        
        logger.info(
            "Model Summary: {}".format(get_model_info(model, self.exp.test_size))
        )

        logger.info("Pruning start...")
        self.prune_model(model)
        ##logger.info("\n{}".format(model))

    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        if self.rank == 0:
            save_model = self.model
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )

    def prune_model(self, model):
        model.cpu()
        
        #Calculate network size before pruning
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("[Total] "+model._get_name()+" Number of Parameters (before ): %.1fM"%(params/1e6))
        
        model.eval()
        num_params_before_pruning = tp.utils.count_params(model)
        # 1. build dependency graph
        strategy = tp.strategy.L1Strategy()
        DG = tp.DependencyGraph()
        out = model(torch.randn([1,3, self.exp.input_size[0], self.exp.input_size[0]]))
        DG.build_dependency(model, example_inputs=torch.randn([1,3, self.exp.input_size[0], self.exp.input_size[1]]))
        # Exclude pruning on output layers
        excluded_layers = list(model.head.cls_preds) + \
                          list(model.head.reg_preds) + \
                          list(model.head.rad_preds) + \
                          list(model.head.lm_preds) + \
                          list(model.head.obj_preds)
        
        #Check total number of sublayers to be pruned
        layer_num = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) and m not in excluded_layers:
                layer_num += 1
        
        #Set prune amount per each layer
        prune_amount_list = np.arange(0.1,self.args.prune_amount,(self.args.prune_amount-0.1)/layer_num)

        layer_cnt = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) and m not in excluded_layers:
                pruning_plan = DG.get_pruning_plan( m, tp.prune_conv, idxs=strategy(m.weight, amount=prune_amount_list[layer_cnt]) )
                print(pruning_plan)
                # execute the plan (prune the model)
                pruning_plan.exec()
                layer_cnt += 1
        num_params_after_pruning = tp.utils.count_params( model )
        print( "  Params: %s => %s"%( num_params_before_pruning, num_params_after_pruning))
        self.model = model
        torch.save(model, 'YOLOX_outputs/'+self.args.experiment_name + '/'+ self.args.prune_level+'_ckpt.pth')
        #self.save_ckpt(self.args.prune_level)

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt

            ckpt = torch.load(ckpt_file, map_location=self.device)
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            # resume the training states variables
            start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
                )
            )  # noqa
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                model = load_ckpt(model, ckpt)
            self.start_epoch = 0

        return model