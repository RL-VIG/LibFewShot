# -*- coding: utf-8 -*-
import datetime
import logging
import os
import builtins
from logging import getLogger
from time import time

import torch
import yaml
from torch import nn
import torch.distributed as dist

from queue import Queue
import core.model as arch
from core.data import get_dataloader
from core.utils import (
    AverageMeter,
    ModelType,
    SaveType,
    TensorboardWriter,
    count_parameters,
    create_dirs,
    get_local_time,
    init_logger_config,
    init_seed,
    prepare_device,
    save_model,
    get_instance,
    data_prefetcher,
    GradualWarmupScheduler,
)


class Trainer(object):
    """
    The trainer.

    Build a trainer from config dict, set up optimizer, model, etc. Train/test/val and log.
    """

    def __init__(self, rank, config):
        self.rank = rank
        self.config = config
        self.config["rank"] = rank
        self.distribute = self.config["n_gpu"] > 1
        (
            self.result_path,
            self.log_path,
            self.checkpoints_path,
            self.viz_path,
        ) = self._init_files(config)
        self.logger = self._init_logger()
        self.device, self.list_ids = self._init_device(rank, config)
        self.writer = self._init_writer(self.viz_path)
        self.train_meter, self.val_meter, self.test_meter = self._init_meter()
        print(self.config)
        self.model, self.model_type = self._init_model(config)
        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
        ) = self._init_dataloader(config)
        (
            self.optimizer,
            self.scheduler,
            self.from_epoch,
            self.best_val_acc,
            self.best_test_acc,
        ) = self._init_optim(config)
        self.val_per_epoch = config["val_per_epoch"]

    def train_loop(self, rank):
        """
        The normal train loop: train-val-test and save model when val-acc increases.
        """
        experiment_begin = time()
        for epoch_idx in range(self.from_epoch + 1, self.config["epoch"]):
            if self.distribute and self.model_type == ModelType.FINETUNING:
                self.train_loader[0].sampler.set_epoch(epoch_idx)
            print("============ Train on the train set ============")
            print("learning rate: {}".format(self.scheduler.get_last_lr()))
            train_acc = self._train(epoch_idx)
            print(" * Acc@1 {:.3f} ".format(train_acc))
            if ((epoch_idx + 1) % self.val_per_epoch) == 0:
                print("============ Validation on the val set ============")
                val_acc = self._validate(epoch_idx, is_test=False)
                print(
                    " * Acc@1 {:.3f} Best acc {:.3f}".format(val_acc, self.best_val_acc)
                )
                print("============ Testing on the test set ============")
                test_acc = self._validate(epoch_idx, is_test=True)
                print(
                    " * Acc@1 {:.3f} Best acc {:.3f}".format(
                        test_acc, self.best_test_acc
                    )
                )
            time_scheduler = self._cal_time_scheduler(experiment_begin, epoch_idx)
            print(" * Time: {}".format(time_scheduler))
            self.scheduler.step()

            if self.rank == 0:
                if ((epoch_idx + 1) % self.val_per_epoch) == 0:
                    if val_acc > self.best_val_acc:
                        self.best_val_acc = val_acc
                        self.best_test_acc = test_acc
                        self._save_model(epoch_idx, SaveType.BEST)

                    if epoch_idx != 0 and epoch_idx % self.config["save_interval"] == 0:
                        self._save_model(epoch_idx, SaveType.NORMAL)

                self._save_model(epoch_idx, SaveType.LAST)

        if self.rank == 0:
            print(
                "End of experiment, took {}".format(
                    str(datetime.timedelta(seconds=int(time() - experiment_begin)))
                )
            )
            print("Result DIR: {}".format(self.result_path))

        if self.writer is not None:
            self.writer.close()
            if self.distribute:
                dist.barrier()
        elif self.distribute:
            dist.barrier()

    def _train(self, epoch_idx):
        """
        The train stage.

        Args:
            epoch_idx (int): Epoch index.

        Returns:
            float: Acc.
        """
        self.model.train()

        meter = self.train_meter
        meter.reset()
        episode_size = (
            1
            if self.model_type == ModelType.FINETUNING
            else self.config["episode_size"]
        )

        end = time()
        log_scale = 1 if self.model_type == ModelType.FINETUNING else episode_size
        for batch_idx, batch in enumerate(zip(*self.train_loader)):
            if self.rank == 0:
                self.writer.set_step(
                    epoch_idx * max(map(len, self.train_loader))
                    + batch_idx * episode_size
                )

            # visualize the weight
            if self.rank == 0 and self.config["log_paramerter"]:
                for i, (name, param) in enumerate(self.model.named_parameters()):
                    if "bn" not in name:
                        save_name = name.replace(".", "/")
                        self.writer.add_histogram(save_name, param)

            meter.update("data_time", time() - end)

            # calculate the output
            calc_begin = time()
            output, acc, loss = self.model(
                [elem for each_batch in batch for elem in each_batch]
            )

            # compute gradients
            self.optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
            # for param in self.model.parameters():
            #     if (param.grad != param.grad).float().sum() != 0:  # nan detected
            #         param.grad.zero_()
            self.optimizer.step()
            meter.update("calc_time", time() - calc_begin)

            # measure accuracy and record loss
            meter.update("loss", loss.item())
            meter.update("acc1", acc)

            # measure elapsed time
            meter.update("batch_time", time() - end)

            # print the intermediate results
            if ((batch_idx + 1) * log_scale % self.config["log_interval"] == 0) or (
                batch_idx + 1
            ) * episode_size >= max(map(len, self.train_loader)) * log_scale:
                info_str = (
                    "Epoch-({}): [{}/{}]\t"
                    "Time {:.3f} ({:.3f})\t"
                    "Calc {:.3f} ({:.3f})\t"
                    "Data {:.3f} ({:.3f})\t"
                    "Loss {:.3f} ({:.3f})\t"
                    "Acc@1 {:.3f} ({:.3f})".format(
                        epoch_idx,
                        (batch_idx + 1) * log_scale,
                        max(map(len, self.train_loader)) * log_scale,
                        meter.last("batch_time"),
                        meter.avg("batch_time"),
                        meter.last("calc_time"),
                        meter.avg("calc_time"),
                        meter.last("data_time"),
                        meter.avg("data_time"),
                        meter.last("loss"),
                        meter.avg("loss"),
                        meter.last("acc1"),
                        meter.avg("acc1"),
                    )
                )
                print(info_str)
            end = time()

        return meter.avg("acc1")

    def _validate(self, epoch_idx, is_test=False):
        """
        The val/test stage.

        Args:
            epoch_idx (int): Epoch index.

        Returns:
            float: Acc.
        """
        # switch to evaluate mode
        self.model.eval()
        if self.distribute:
            self.model.module.reverse_setting_info()
        else:
            self.model.reverse_setting_info()
        meter = self.test_meter if is_test else self.val_meter
        meter.reset()
        episode_size = self.config["episode_size"]

        end = time()
        enable_grad = self.model_type != ModelType.METRIC
        log_scale = self.config["episode_size"]
        with torch.set_grad_enabled(enable_grad):
            loader = self.test_loader if is_test else self.val_loader
            for batch_idx, batch in enumerate(zip(*loader)):
                if self.rank == 0:
                    self.writer.set_step(
                        int(
                            (
                                epoch_idx * max(map(len, loader))
                                + batch_idx * episode_size
                            )
                            * self.config["tb_scale"]
                        )
                    )

                meter.update("data_time", time() - end)

                # calculate the output
                calc_begin = time()
                output, acc = self.model(
                    [elem for each_batch in batch for elem in each_batch]
                )
                meter.update("calc_time", time() - calc_begin)

                # measure accuracy and record loss
                meter.update("acc1", acc)

                # measure elapsed time
                meter.update("batch_time", time() - end)

                if ((batch_idx + 1) * log_scale % self.config["log_interval"] == 0) or (
                    batch_idx + 1
                ) * episode_size >= max(map(len, loader)) * log_scale:
                    info_str = (
                        "Epoch-({}): [{}/{}]\t"
                        "Time {:.3f} ({:.3f})\t"
                        "Calc {:.3f} ({:.3f})\t"
                        "Data {:.3f} ({:.3f})\t"
                        "Acc@1 {:.3f} ({:.3f})".format(
                            epoch_idx,
                            (batch_idx + 1) * log_scale,
                            max(map(len, loader)) * log_scale,
                            meter.last("batch_time"),
                            meter.avg("batch_time"),
                            meter.last("calc_time"),
                            meter.avg("calc_time"),
                            meter.last("data_time"),
                            meter.avg("data_time"),
                            meter.last("acc1"),
                            meter.avg("acc1"),
                        )
                    )
                    print(info_str)
                end = time()

        if self.distribute:
            self.model.module.reverse_setting_info()
        else:
            self.model.reverse_setting_info()
        return meter.avg("acc1")

    def _init_files(self, config):
        """
        Init result_path(checkpoints_path, log_path, viz_path) from the config dict.

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of (result_path, log_path, checkpoints_path, viz_path).
        """
        if self.config["resume"]:
            result_path = self.config["resume_path"]
            checkpoints_path = os.path.join(result_path, "checkpoints")
            log_path = os.path.join(result_path, "log_files")
            viz_path = os.path.join(log_path, "tfboard_files")
        else:
            # you should ensure that data_root name contains its true name
            base_dir = "{}-{}-{}-{}-{}".format(
                config["classifier"]["name"],
                config["data_root"].split("/")[-1],
                config["backbone"]["name"],
                config["way_num"],
                config["shot_num"],
            )
            result_dir = (
                base_dir
                + "{}-{}".format(
                    ("-" + config["tag"]) if config["tag"] is not None else "",
                    get_local_time(),
                )
                if config["log_name"] is None
                else config["log_name"]
            )
            result_path = os.path.join(config["result_root"], result_dir)
            # print("Result DIR: " + result_path)
            checkpoints_path = os.path.join(result_path, "checkpoints")
            log_path = os.path.join(result_path, "log_files")
            viz_path = os.path.join(log_path, "tfboard_files")
            if self.rank == 0:
                create_dirs([result_path, log_path, checkpoints_path, viz_path])

                with open(
                    os.path.join(result_path, "config.yaml"), "w", encoding="utf-8"
                ) as fout:
                    fout.write(yaml.dump(config))

        init_logger_config(
            config["log_level"],
            log_path,
            config["classifier"]["name"],
            config["backbone"]["name"],
            rank=self.rank,
        )

        return result_path, log_path, checkpoints_path, viz_path

    def _init_logger(self):
        self.logger = getLogger(__name__)

        # hack print
        def use_logger(
            *msg, level="info", all_rank=False
        ):  # if neccessary, you need to specific the `level` and `all_rank` params
            if self.rank != 0 and not all_rank:
                return
            try:
                for m in msg:
                    if self.rank == 0:
                        getattr(self.logger, level)(m)
                    else:
                        getattr(logging, level)(m)
            except os.error:
                raise ("logging have no {} level".format(level))

        builtins.print = use_logger

        return self.logger

    def _init_dataloader(self, config):
        """
        Init dataloaders.(train_loader, val_loader and test_loader)

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of (train_loader, val_loader and test_loader).
        """
        self._check_data_config()
        distribute = self.distribute
        train_loader = get_dataloader(config, "train", self.model_type, distribute)
        val_loader = get_dataloader(config, "val", self.model_type, distribute)
        test_loader = get_dataloader(config, "test", self.model_type, distribute)

        return train_loader, val_loader, test_loader

    def _init_model(self, config):
        """
        Init model(backbone+classifier) from the config dict and load the pretrained params or resume from a
        checkpoint, then parallel if necessary .

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of the model and model's type.
        """
        emb_func = get_instance(arch, "backbone", config)
        model_kwargs = {
            "way_num": config["way_num"],
            "shot_num": config["shot_num"] * config["augment_times"],
            "query_num": config["query_num"],
            "test_way": config["test_way"],
            "test_shot": config["test_shot"] * config["augment_times"],
            "test_query": config["test_query"],
            "emb_func": emb_func,
            "device": self.device,
        }
        model = get_instance(arch, "classifier", config, **model_kwargs)

        print(model)
        print("Trainable params in the model: {}".format(count_parameters(model)))
        # FIXME: May be inaccurate

        if self.config["pretrain_path"] is not None:
            print(
                "load pretraining emb_func from {}".format(self.config["pretrain_path"])
            )
            state_dict = torch.load(self.config["pretrain_path"], map_location="cpu")
            msg = model.emb_func.load_state_dict(state_dict, strict=False)

            if len(msg.missing_keys) != 0:
                print("Missing keys:{}".format(msg.missing_keys), level="warning")
            if len(msg.unexpected_keys) != 0:
                print("Unexpected keys:{}".format(msg.unexpected_keys), level="warning")

        if self.config["resume"]:
            resume_path = os.path.join(
                self.config["resume_path"], "checkpoints", "model_last.pth"
            )
            print("load the resume model checkpoints dict from {}.".format(resume_path))
            state_dict = torch.load(resume_path, map_location="cpu")["model"]
            msg = model.load_state_dict(state_dict, strict=False)

            if len(msg.missing_keys) != 0:
                print("missing keys:{}".format(msg.missing_keys), level="warning")
            if len(msg.unexpected_keys) != 0:
                print("unexpected keys:{}".format(msg.unexpected_keys), level="warning")

        if self.distribute:
            # higher order grad of BN in multi gpu will conflict with syncBN
            # FIXME MAML with multi GPU conflict with syncBN
            if not (
                self.config["classifier"]["name"] in ["MAML"]
                and self.config["n_gpu"] > 1
            ):
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            else:
                print(
                    "{} with multi GPU will conflict with syncBN".format(
                        self.config["classifier"]["name"]
                    ),
                    level="warning",
                )
            model = model.to(self.rank)
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=True,
            )

            return model, model.module.model_type
        else:
            model = model.to(self.rank)

            return model, model.model_type

    def _init_optim(self, config):
        """
        Init the optimizers and scheduler from config, if necessary, load the state dict from a checkpoint.

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of optimizer, scheduler and epoch_index.
        """
        params_idx = []
        params_dict_list = []
        if "other" in config["optimizer"] and config["optimizer"]["other"] is not None:
            for key, value in config["optimizer"]["other"].items():
                # FIXME: Set the learning rate for the model specified parameter, including nn.Module and nn.Parameter
                # Original:
                # Pre-modified:
                # if self.distribute:
                #     sub_model = getattr(self.model.module, key)
                # else:
                #     sub_model = getattr(self.model, key)

                # Modified:
                if self.distribute:
                    sub_model = eval("self.model.module." + key)
                else:
                    sub_model = eval("self.model." + key)
                if isinstance(sub_model, nn.Module):
                    sub_model_params_list = list(sub_model.parameters())
                elif isinstance(sub_model, nn.Parameter):
                    sub_model_params_list = [sub_model]
                else:
                    raise Exception("Unrecognized type in optimizer.other")

                params_idx.extend(list(map(id, sub_model_params_list)))
                if value is None:
                    for p in sub_model_params_list:
                        p.requires_grad = False
                else:
                    param_dict = {"params": sub_model_params_list}
                    if isinstance(value, float):
                        param_dict.update({"lr": value})
                    elif isinstance(value, dict):
                        param_dict.update(value)
                    else:
                        raise Exception("Wrong config in optimizer.other")
                    params_dict_list.append(param_dict)

        params_dict_list.append(
            {
                "params": filter(
                    lambda p: id(p) not in params_idx, self.model.parameters()
                )
            }
        )
        optimizer = get_instance(
            torch.optim, "optimizer", config, params=params_dict_list
        )
        scheduler = GradualWarmupScheduler(
            optimizer, self.config
        )  # if config['warmup']==0, scheduler will be a normal lr_scheduler, jump into this class for details
        print(optimizer)
        from_epoch = -1
        best_val_acc = float("-inf")
        best_test_acc = float("-inf")
        if self.config["resume"]:
            resume_path = os.path.join(
                self.config["resume_path"], "checkpoints", "model_last.pth"
            )
            print(
                "load the optimizer, lr_scheduler and epoch checkpoints dict from {}.".format(
                    resume_path
                )
            )
            all_state_dict = torch.load(resume_path, map_location="cpu")
            state_dict = all_state_dict["optimizer"]
            optimizer.load_state_dict(state_dict)
            state_dict = all_state_dict["lr_scheduler"]
            scheduler.load_state_dict(state_dict)
            from_epoch = all_state_dict["epoch"]
            best_val_acc = all_state_dict["best_val_acc"]
            best_test_acc = all_state_dict["best_val_acc"]
            print("model resume from the epoch {}".format(from_epoch))

        return optimizer, scheduler, from_epoch, best_val_acc, best_test_acc

    def _init_device(self, rank, config):
        """
        Init the devices from the config file.

        Args:
            config (dict): Parsed config file.

        Returns:
            tuple: A tuple of deviceand list_ids.
        """
        init_seed(config["seed"], config["deterministic"])
        device, list_ids = prepare_device(
            rank,
            config["device_ids"],
            config["n_gpu"],
            backend="nccl"
            if "dist_backend" not in self.config
            else self.config["dist_backend"],
            dist_url="tcp://127.0.0.1:" + str(config["port"])
            if "dist_url" not in self.config
            else self.config["dist_url"],
        )
        torch.cuda.set_device(self.rank)

        return device, list_ids

    def _save_model(self, epoch, save_type=SaveType.NORMAL):
        """
        Save the model, optimizer, scheduler and epoch.

        TODO

        Args:
            epoch (int): the current epoch index.
            save_type (SaveType, optional): type of (last, best). Defaults to SaveType.NORMAL.
        """
        save_model(
            self.model,
            self.optimizer,
            self.scheduler,
            self.checkpoints_path,
            "model",
            epoch,
            self.best_val_acc,
            self.best_test_acc,
            save_type,
            len(self.list_ids) > 1,
        )

        if save_type != SaveType.LAST:
            save_list = self.config["save_part"]
            if save_list is not None:
                for save_part in save_list:
                    save_module = self.model.module if self.distribute else self.model
                    if hasattr(save_module, save_part):
                        save_model(
                            getattr(save_module, save_part),
                            self.optimizer,
                            self.scheduler,
                            self.checkpoints_path,
                            save_part,
                            epoch,
                            self.best_val_acc,
                            self.best_test_acc,
                            save_type,
                            len(self.list_ids) > 1,
                        )
                    else:
                        print(
                            "{} is not included in {}".format(
                                save_part, self.config["classifier"]["name"]
                            ),
                            level="warning",
                        )

    def _init_meter(self):
        """
        Init the AverageMeter of train/val/test stage to cal avg... of batch_time, data_time,calc_time ,loss and acc1.

        Returns:
            tuple: A tuple of train_meter, val_meter, test_meter.
        """
        train_meter = AverageMeter(
            "train",
            ["batch_time", "data_time", "calc_time", "loss", "acc1"],
            self.writer,
        )
        val_meter = AverageMeter(
            "val",
            ["batch_time", "data_time", "calc_time", "acc1"],
            self.writer,
        )
        test_meter = AverageMeter(
            "test",
            ["batch_time", "data_time", "calc_time", "acc1"],
            self.writer,
        )

        return train_meter, val_meter, test_meter

    def _init_writer(self, viz_path):
        """
        Init the tensorboard writer.

        Return:
            writer: tensorboard writer
        """
        if self.rank == 0:
            writer = TensorboardWriter(viz_path)
            return writer
        else:
            return None

    def _check_data_config(self):
        """
        Check the config params.
        """
        # check: episode_size >= n_gpu and episode_size != 0
        assert (
            self.config["episode_size"] >= self.config["n_gpu"]
            and self.config["episode_size"] != 0
        ), "episode_size {} should be >= n_gpu {} and != 0".format(
            self.config["episode_size"], self.config["n_gpu"]
        )

        # check: episode_size % n_gpu == 0
        assert (
            self.config["episode_size"] % self.config["n_gpu"] == 0
        ), "episode_size {} % n_gpu {} != 0".format(
            self.config["episode_size"], self.config["n_gpu"]
        )

        # check: episode_num % episode_size == 0
        assert (
            self.config["train_episode"] % self.config["episode_size"] == 0
        ), "train_episode {} % episode_size  {} != 0".format(
            self.config["train_episode"], self.config["episode_size"]
        )

        assert (
            self.config["test_episode"] % self.config["episode_size"] == 0
        ), "test_episode {} % episode_size  {} != 0".format(
            self.config["test_episode"], self.config["episode_size"]
        )

    def _cal_time_scheduler(self, start_time, epoch_idx):
        """
        Calculate the remaining time and consuming time of the training process.

        Returns:
            str: A string similar to "00:00:00/0 days, 00:00:00". First: comsuming time; Second: total time.
        """
        total_epoch = self.config["epoch"] - self.from_epoch - 1
        now_epoch = epoch_idx - self.from_epoch

        time_consum = datetime.datetime.now() - datetime.datetime.fromtimestamp(
            start_time
        )
        time_consum -= datetime.timedelta(microseconds=time_consum.microseconds)
        time_remain = (time_consum * (total_epoch - now_epoch)) / (now_epoch)

        res_str = str(time_consum) + "/" + str(time_remain + time_consum)

        return res_str
