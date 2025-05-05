import os
import json
import random
import string
import logging
import torch
import time
import math
import threading
import botocore.exceptions
import pandas as pd
from functools import partial
from datetime import datetime

from .dataset_s3 import DatasetS3
from .dataset_vcmi import DatasetVCMI
from .misc import TableColumn, dig, aggregate_metrics, timer_stats, safe_mean
from .persistence import load_local_or_s3_checkpoint, save_checkpoint, save_buffer_async
from .stats import Stats
from .structured_logger import StructuredLogger
from .timer import Timer
from .wandb import setup_wandb
from .weights import build_feature_weights

from ..util.constants_v12 import (
    STATE_SIZE_GLOBAL,
    STATE_SIZE_ONE_PLAYER,
    STATE_SIZE_ONE_HEX,
    N_ACTIONS,
    GLOBAL_ATTR_MAP,
    PLAYER_ATTR_MAP,
    HEX_ATTR_MAP,
)


DIM_OTHER = STATE_SIZE_GLOBAL + 2*STATE_SIZE_ONE_PLAYER
DIM_HEXES = 165*STATE_SIZE_ONE_HEX
DIM_OBS = DIM_OTHER + DIM_HEXES


def train(
    config,
    resume_config,
    loglevel,
    dry_run,
    no_wandb,
    sample_only,
    model_creator,
    buffer_creator,
    vcmi_dataloader_functor,
    s3_dataloader_functor,
    eval_model_fn,
    train_model_fn
):
    if resume_config:
        with open(resume_config, "r") as f:
            print(f"Resuming from config: {f.name}")
            config = json.load(f)

        run_id = config["run"]["id"]
        config["run"]["resumed_config"] = resume_config
    else:
        run_id = ''.join(random.choices(string.ascii_lowercase, k=8))
        config["run"] = dict(
            id=run_id,
            name=config["name_template"].format(id=run_id, datetime=datetime.utcnow().strftime("%Y%m%d_%H%M%S")),
            out_dir=os.path.abspath(config["out_dir_template"].format(id=run_id)),
            resumed_config=None,
        )

    checkpoint_s3_config = dig(config, "s3", "checkpoint")
    train_s3_config = dig(config, "s3", "data", "train")
    eval_s3_config = dig(config, "s3", "data", "eval")
    train_env_config = dig(config, "env", "train")
    eval_env_config = dig(config, "env", "eval")

    train_sample_from_env = train_env_config is not None
    eval_sample_from_env = eval_env_config is not None

    train_sample_from_s3 = (not train_sample_from_env) and train_s3_config is not None
    eval_sample_from_s3 = (not eval_sample_from_env) and eval_s3_config is not None

    train_save_samples = train_sample_from_env and train_s3_config is not None
    eval_save_samples = eval_sample_from_env and eval_s3_config is not None

    train_batch_size = config["train"]["batch_size"]
    eval_batch_size = config["eval"]["batch_size"]

    if train_env_config:
        # Prevent guaranteed waiting time for each batch during training
        assert train_batch_size <= (train_env_config["num_workers"] * train_env_config["batch_size"])
        # Samples would be lost otherwise (batched_iter uses loop with step=batch_size)
        assert (train_env_config["num_workers"] * train_env_config["batch_size"]) % train_batch_size == 0
    else:
        assert train_batch_size <= (train_s3_config["num_workers"] * train_s3_config["batch_size"])
        assert (train_s3_config["num_workers"] * train_s3_config["batch_size"]) % train_batch_size == 0

    if eval_env_config:
        # Samples would be lost otherwise (batched_iter uses loop with step=batch_size)
        assert eval_batch_size <= (eval_env_config["num_workers"] * eval_env_config["batch_size"])
        assert (eval_env_config["num_workers"] * eval_env_config["batch_size"]) % eval_batch_size == 0
    else:
        assert eval_batch_size <= (eval_s3_config["num_workers"] * eval_s3_config["batch_size"])
        assert (eval_s3_config["num_workers"] * eval_s3_config["batch_size"]) % eval_batch_size == 0

    assert config["checkpoint_interval_s"] > config["eval"]["interval_s"]
    assert config["permanent_checkpoint_interval_s"] > config["eval"]["interval_s"]

    # assert config["wandb_log_interval_s"] > config["eval"]["interval_s"]

    assert config["wandb_table_log_interval_s"] >= config["wandb_table_update_interval_s"]
    assert config["wandb_table_update_interval_s"] >= config["eval"]["interval_s"]

    if config["wandb_table_log"]:
        # Every update addds `num_attrs` rows
        num_updates = config["wandb_table_log_interval_s"] // config["wandb_table_update_interval_s"]
        num_attrs = sum(len(attrmap) for attrmap in [GLOBAL_ATTR_MAP, PLAYER_ATTR_MAP, HEX_ATTR_MAP])
        max_rows = 200_000  # can be changed via wandb.Table.MAX_ARTIFACT_ROWS = ...
        max_updates = max_rows // num_attrs
        assert num_updates < max_updates

    os.makedirs(config["run"]["out_dir"], exist_ok=True)

    with open(os.path.join(config["run"]["out_dir"], f"{run_id}-config.json"), "w") as f:
        print(f"Saving new config to: {f.name}")
        json.dump(config, f, indent=4)

    logger = StructuredLogger(level=getattr(logging, loglevel), filename=os.path.join(config["run"]["out_dir"], f"{run_id}.log"), context=dict(run_id=run_id))
    logger.info(dict(config=config))

    learning_rate = config["train"]["learning_rate"]
    train_epochs = config["train"]["epochs"]

    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/6
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_creator(device=device)
    feature_weights = build_feature_weights(model, config["weights"])

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if device.type == "cuda":
        scaler = torch.amp.GradScaler()
    else:
        scaler = None

    optimize_local_storage = config.get("s3", {}).get("optimize_local_storage")

    def make_vcmi_dataloader(cfg, mq):
        return torch.utils.data.DataLoader(
            DatasetVCMI(logger=logger, env_kwargs=cfg["kwargs"], metric_queue=mq, mw_functor=vcmi_dataloader_functor),
            batch_size=cfg["batch_size"],
            num_workers=cfg["num_workers"],
            prefetch_factor=cfg["prefetch_factor"],
            # persistent_workers=True,  # no effect here
        )

    def make_s3_dataloader(cfg, mq, split_ratio=None, split_side=None):
        return torch.utils.data.DataLoader(
            DatasetS3(
                logger=logger,
                bucket_name=cfg["bucket_name"],
                s3_dir=cfg["s3_dir"],
                cache_dir=cfg["cache_dir"],
                cached_files_max=cfg["cached_files_max"],
                shuffle=cfg["shuffle"],
                split_ratio=split_ratio,
                split_side=split_side,
                aws_access_key=os.environ["AWS_ACCESS_KEY"],
                aws_secret_key=os.environ["AWS_SECRET_KEY"],
                metric_queue=mq,
                # mw_functor=
            ),
            batch_size=cfg["batch_size"],
            num_workers=cfg["num_workers"],
            prefetch_factor=cfg["prefetch_factor"],
            pin_memory=cfg["pin_memory"]
        )

    train_metric_queue = torch.multiprocessing.Queue()
    eval_metric_queue = torch.multiprocessing.Queue()

    if train_sample_from_env:
        dataloader_obj = make_vcmi_dataloader(train_env_config, train_metric_queue)
    if eval_sample_from_env:
        eval_dataloader_obj = make_vcmi_dataloader(eval_env_config, eval_metric_queue)
    if train_sample_from_s3:
        dataloader_obj = make_s3_dataloader(train_s3_config, train_metric_queue, 0.98, 0)
    if eval_sample_from_s3:
        # eval_dataloader_obj = make_s3_dataloader(eval_s3_config, eval_metric_queue)
        eval_dataloader_obj = make_s3_dataloader(dict(eval_s3_config, s3_dir=train_s3_config["s3_dir"]), eval_metric_queue, 0.98, 1)

    def make_buffer(dloader, name):
        return buffer_creator(logger=logger, dataloader=dloader, dim_obs=DIM_OBS, n_actions=N_ACTIONS, name=name, device=device)

    buffer = make_buffer(dataloader_obj, "train")
    dataloader = iter(dataloader_obj)
    eval_buffer = make_buffer(eval_dataloader_obj, "eval")
    eval_dataloader = iter(eval_dataloader_obj)
    stats = Stats(model, device=device)

    if resume_config:
        load_checkpoint = partial(
            load_local_or_s3_checkpoint,
            logger,
            dry_run,
            checkpoint_s3_config,
            optimize_local_storage,
            device,
            config["run"]["out_dir"],
            run_id,
        )

        load_checkpoint("model", model, strict=True)
        load_checkpoint("optimizer", optimizer)
        optimizer.param_groups[0]["lr"] = learning_rate

        if scaler:
            try:
                load_checkpoint("scaler", scaler)
            except botocore.exceptions.ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    logger.warn("WARNING: scaler weights not found (maybe the model was trained on CPU only?)")
                else:
                    raise

    if no_wandb:
        from unittest.mock import Mock
        wandb = Mock()
    else:
        wandb = setup_wandb(config, model, __file__)

    accumulated_logs = {}

    def accumulate_logs(data):
        for k, v in data.items():
            if k not in accumulated_logs:
                accumulated_logs[k] = [v]
            else:
                accumulated_logs[k].append(v)

    def aggregate_logs():
        agg_data = {k: safe_mean(v) for k, v in accumulated_logs.items()}
        accumulated_logs.clear()
        return agg_data

    # Aggregate loss for entire stage (1 row per context/datatype pair)
    def aggregate_losses(stage, df):
        aggregated = df.groupby([TableColumn.CONTEXT, TableColumn.DATATYPE])[TableColumn.LOSS].sum()
        # => keys = tuple(context, datatype), values = float(loss)

        res = {f"{stage}_loss/total": 0}
        for (ctx, dt), v in aggregated.items():
            res[f"{stage}_loss/{ctx}/{dt}"] = v
            res[f"{stage}_loss/total"] += v

        return res

    table_columns = [
        TableColumn.STEP,
        TableColumn.STAGE,
        TableColumn.ATTRIBUTE,
        TableColumn.CONTEXT,
        TableColumn.DATATYPE,
        TableColumn.LOSS,
    ]

    # Accumulates data for table update
    df_stages = pd.DataFrame(columns=table_columns)

    # Accumulates data for wandb.log
    df_wandb = pd.DataFrame(columns=table_columns)

    # Shorthand to supress pandas warning
    concat_dfs = lambda dfa, dfb: dfb.copy() if dfa.empty else pd.concat([dfa, dfb], ignore_index=True)
    reset_df = lambda df: df.iloc[0:0]

    wandb.log({
        "train/learning_rate": optimizer.param_groups[0]["lr"],
        "train/buffer_capacity": buffer.capacity,
        "train/epochs": train_epochs,
        "train/batch_size": train_batch_size,
        "eval/buffer_capacity": eval_buffer.capacity,
        "eval/batch_size": eval_batch_size,
    }, commit=False)

    last_checkpoint_at = time.time()
    last_permanent_checkpoint_at = time.time()
    last_evaluation_at = 0
    last_wandb_commit_log_at = time.time()
    last_wandb_table_update_at = time.time()
    last_wandb_table_log_at = time.time()

    # during training, we simply check if the event is set and optionally skip the upload
    # Non-bloking, but uploads may be skipped (checkpoint uploads)
    uploading_event = threading.Event()
    train_uploading_event_buf = threading.Event()
    eval_uploading_event_buf = threading.Event()

    # during sample collection, we use a cond lock to prevent more than 1 upload at a time
    # Blocking, but all uploads are processed (buffer uploads)
    train_uploading_cond = threading.Condition()
    eval_uploading_cond = threading.Condition()

    timers = {
        "all": Timer(),
        "sample": Timer(),
        "train": Timer(),
        "eval": Timer(),
    }

    timers["all"].start()

    eval_loss_best = None

    while True:
        now = time.time()

        with timers["sample"]:
            buffer.load_samples(dataloader)

        logger.info("Samples loaded: %d" % buffer.capacity)

        assert buffer.full and not buffer.index

        if train_save_samples:
            save_buffer_async(
                run_id=run_id,
                logger=logger,
                dry_run=dry_run,
                buffer=buffer,
                env_config=train_env_config,
                s3_config=train_s3_config,
                allow_skip=not sample_only,
                uploading_cond=train_uploading_cond,
                uploading_event_buf=train_uploading_event_buf,
                optimize_local_storage=optimize_local_storage
            )

        if sample_only:
            stats.iteration += 1
            continue

        # loss_weights = stats.compute_loss_weights()

        wlog = {"iteration": stats.iteration}

        # Evaluate first (for a baseline when resuming with modified params)
        if now - last_evaluation_at > config["eval"]["interval_s"]:
            last_evaluation_at = now

            with timers["sample"]:
                eval_buffer.load_samples(eval_dataloader)

            with timers["eval"]:
                df_stage_eval, eval_total_wait = eval_model_fn(
                    logger=logger,
                    model=model,
                    loss_weights=feature_weights,
                    buffer=eval_buffer,
                    batch_size=eval_batch_size,
                )

            # Mean loss for entire stage (1 row per attribute)
            df_stage_eval[TableColumn.STAGE] = "eval"
            df_stages = concat_dfs(df_stages, df_stage_eval)

            wlog.update(**aggregate_losses("eval", df_stage_eval))
            wlog["eval_dataset/wait_time_s"] = eval_total_wait

            train_dataset_metrics = aggregate_metrics(train_metric_queue)
            if train_dataset_metrics:
                wlog["train_dataset/avg_worker_utilization"] = train_dataset_metrics

            eval_dataset_metrics = aggregate_metrics(eval_metric_queue)
            if eval_dataset_metrics:
                wlog["eval_dataset/avg_worker_utilization"] = eval_dataset_metrics

            eval_loss = wlog["eval_loss/total"]

            if eval_save_samples:
                save_buffer_async(
                    run_id=run_id,
                    logger=logger,
                    dry_run=dry_run,
                    buffer=eval_buffer,
                    env_config=eval_env_config,
                    s3_config=eval_s3_config,
                    allow_skip=not sample_only,
                    uploading_cond=eval_uploading_cond,
                    uploading_event_buf=eval_uploading_event_buf,
                    optimize_local_storage=optimize_local_storage
                )

            if now - last_checkpoint_at > config["checkpoint_interval_s"]:
                last_checkpoint_at = now

                if eval_loss_best is None:
                    # Initial baseline for resumed configs
                    eval_loss_best = eval_loss
                    logger.info("No baseline for checkpoint yet (eval_loss=%f, eval_loss_best=None), setting it now" % (eval_loss))
                elif eval_loss and (math.isnan(eval_loss) or eval_loss >= eval_loss_best):
                    logger.info("Bad checkpoint (eval_loss=%f, eval_loss_best=%f), will skip it" % (eval_loss, eval_loss_best))
                else:
                    logger.info("Good checkpoint (eval_loss=%f, eval_loss_best=%f), will save it" % (eval_loss, eval_loss_best))
                    eval_loss_best = eval_loss
                    thread = threading.Thread(target=save_checkpoint, kwargs=dict(
                        logger=logger,
                        dry_run=dry_run,
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        out_dir=config["run"]["out_dir"],
                        run_id=run_id,
                        optimize_local_storage=optimize_local_storage,
                        s3_config=config.get("s3", {}).get("checkpoint"),
                        uploading_event=uploading_event
                    ))
                    thread.start()

            if now - last_permanent_checkpoint_at > config["permanent_checkpoint_interval_s"]:
                last_permanent_checkpoint_at = now
                logger.info("Time for a permanent checkpoint")
                thread = threading.Thread(target=save_checkpoint, kwargs=dict(
                    logger=logger,
                    dry_run=dry_run,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    out_dir=config["run"]["out_dir"],
                    run_id=run_id,
                    optimize_local_storage=optimize_local_storage,
                    s3_config=config.get("s3", {}).get("checkpoint"),
                    uploading_event=threading.Event(),  # don't skip this upload
                    permanent=True,
                    config=config,
                ))
                thread.start()

        with timers["train"]:
            df_stage_train, train_total_wait = train_model_fn(
                logger=logger,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                buffer=buffer,
                stats=stats,
                loss_weights=feature_weights,
                epochs=train_epochs,
                batch_size=train_batch_size,
                accumulate_grad=config["train"]["accumulate_grad"],
            )

        # Mean loss for entire stage (1 row per attribute)
        df_stage_train[TableColumn.STAGE] = "train"
        df_stages = concat_dfs(df_stages, df_stage_train)

        wlog.update(**aggregate_losses("train", df_stage_train))
        wlog["train_dataset/wait_time_s"] = train_total_wait

        # from .analyze_loss import analyze_loss
        # topk = 5
        # analyzed_attrs, topk_frac, top3_loss = analyze_loss(model, train_attrlosses, eval_attrlosses, topk=topk)
        # for group_name, group_topk_attrs in topk_frac.items():
        #     print("Top%d loss diff attrs in group: %s" % (topk, group_name))
        #     for name, (diff_loss, diff_frac, eval_loss, train_loss) in group_topk_attrs:
        #         print("    %-30s diff: %.3f (x%.1f) | eval: %.3f | train: %.3f" % (name, diff_loss, diff_frac, eval_loss, train_loss))
        # import ipdb; ipdb.set_trace()  # noqa

        accumulate_logs(wlog)

        if now - last_wandb_commit_log_at > config["wandb_log_interval_s"]:
            last_wandb_commit_log_at = now

            if now - last_wandb_table_update_at > config["wandb_table_update_interval_s"]:
                last_wandb_table_update_at = now

                # Aggregate loss for the wandb log update
                # (1 row per attribute)
                df_update = df_stages.groupby([
                    TableColumn.STAGE,
                    TableColumn.ATTRIBUTE,
                    TableColumn.CONTEXT,
                    TableColumn.DATATYPE,
                ], as_index=False)[TableColumn.LOSS].mean()

                df_update[TableColumn.STEP] = wandb.run.step

                # XXX: Cannot log the wtable via structured logger (fails to serialize it)
                # => log it separately
                logger.info({"event": "Attribute losses", "tables": {
                    "train": dict(df_update[df_update[TableColumn.STAGE] == "train"][[TableColumn.ATTRIBUTE, TableColumn.LOSS]].values),
                    "eval": dict(df_update[df_update[TableColumn.STAGE] == "eval"][[TableColumn.ATTRIBUTE, TableColumn.LOSS]].values),
                }})

                df_wandb = concat_dfs(df_wandb, df_update)

                if now - last_wandb_table_log_at > config["wandb_table_log_interval_s"]:
                    last_wandb_table_log_at = now
                    if config["wandb_table_log"]:
                        logger.info("Uploading table with %d rows to wandb..." % len(df_wandb))
                        wandb.log({"tables/loss": wandb.Table(dataframe=df_wandb)}, commit=False)
                    df_wandb = reset_df(df_wandb)

                df_update = reset_df(df_update)
                df_stages = reset_df(df_stages)

            wlog.update(aggregate_logs())
            wlog.update(timer_stats(timers))
            wandb.log(wlog, commit=True)

        logger.info(wlog)

        # XXX: must log timers here (some may have been skipped)
        stats.iteration += 1
