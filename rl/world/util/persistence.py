import os
import pathlib
import shutil
import torch
import numpy as np
import time
import threading
import json
import copy
import boto3
import botocore.exceptions
from boto3.s3.transfer import TransferConfig


def init_s3_client():
    return boto3.client(
        's3',
        aws_access_key_id=os.environ["AWS_ACCESS_KEY"],
        aws_secret_access_key=os.environ["AWS_SECRET_KEY"],
        region_name="eu-north-1",
        config=botocore.config.Config(connect_timeout=10, read_timeout=30)
    )


def _s3_download(logger, dry_run, s3_config, filename):
    if os.path.exists(f"{filename}~"):
        if os.path.exists(filename):
            msg = f"Lockfile for {filename} still exists => deleting local (corrupted) file"
            if dry_run:
                logger.warn(f"{msg} (--dry-run)")
            else:
                logger.warn(msg)
                os.unlink(filename)
        if not dry_run:
            os.unlink(f"{filename}~")

    # Download is OK even if --dry-run is given (nothing overwritten)
    if s3_config and not os.path.exists(filename):
        logger.debug("Local file does not exist, try S3")

        s3_filename = f"{s3_config['s3_dir']}/{os.path.basename(filename)}"
        logger.info(f"Download s3://{s3_config['bucket_name']}/{s3_filename} ...")

        if os.path.exists(f"{filename}.tmp"):
            os.unlink(f"{filename}.tmp")
        init_s3_client().download_file(s3_config["bucket_name"], s3_filename, f"{filename}.tmp")
        shutil.move(f"{filename}.tmp", filename)


def _cleanup(logger, dry_run, optimize_local_storage, filename):
    if not dry_run and not optimize_local_storage:
        backname = "%s-%d.pt" % (filename.removesuffix(".pt"), time.time())
        logger.debug(f"Backup resumed model weights as {backname}")
        shutil.copy2(filename, backname)


# Merge b into a, optionally preventing new keys; does not mutate inputs
def deepmerge(a: dict, b: dict, in_place=False, allow_new=True, update_existing=True, path=[]):
    if len(path) == 0 and not in_place:
        a = copy.deepcopy(a)

    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                deepmerge(a[key], b[key], in_place, allow_new, update_existing, path + [str(key)])
            elif update_existing and a[key] != b[key]:
                a[key] = b[key]
        elif allow_new:
            a[key] = b[key]
        else:
            raise KeyError(key)
    return a


def load_checkpoint(
    logger,
    dry_run,
    model,
    optimizer,
    scaler,
    out_dir,
    run_id,
    optimize_local_storage,
    s3_config,
    device,
    state=None,  # object with .to_json() and .from_json(string)
):
    prefix = run_id
    f_model = os.path.join(out_dir, f"{prefix}-model.pt")
    f_optimizer = os.path.join(out_dir, f"{prefix}-optimizer.pt")
    f_scaler = os.path.join(out_dir, f"{prefix}-scaler.pt")
    f_state = os.path.join(out_dir, f"{prefix}-state.json")

    logger.info(dict(
        event="Loading checkpoint...",
        model=f_model,
        optimizer=f_optimizer,
        scaler=f_scaler,
        state=f_state,
    ))

    _s3_download(logger, dry_run, s3_config, f_model)
    model.load_state_dict(torch.load(f_model, weights_only=True, map_location=device), strict=True)
    _cleanup(logger, dry_run, optimize_local_storage, f_model)

    _s3_download(logger, dry_run, s3_config, f_optimizer)
    optimizer.load_state_dict(torch.load(f_optimizer, weights_only=True, map_location=device))
    _cleanup(logger, dry_run, optimize_local_storage, f_optimizer)

    if scaler:
        try:
            _s3_download(logger, dry_run, s3_config, f_scaler)
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                logger.warn("WARNING: scaler weights not found (maybe the model was trained on CPU only?)")
            else:
                raise
        scaler.load_state_dict(torch.load(f_scaler, weights_only=True, map_location=device))
        _cleanup(logger, dry_run, optimize_local_storage, f_scaler)

    if state:
        assert callable(state.to_json) and callable(state.from_json)
        _s3_download(logger, dry_run, s3_config, f_state)
        with open(f_state, "r") as f:
            state.from_json(f.read())
        _cleanup(logger, dry_run, optimize_local_storage, f_state)


def save_checkpoint(
    logger,
    dry_run,
    model,
    optimizer,
    scaler,
    state,  # object with .to_json() and .from_json(string)
    out_dir,
    run_id,
    optimize_local_storage,
    s3_config,
    uploading_event,
    permanent=False,
    config=None,
):
    if permanent:
        assert config, "config is also needed for permanent checkpoints"
        prefix = f"{run_id}-{time.time():.0f}"
    else:
        prefix = run_id

    f_model = os.path.join(out_dir, f"{prefix}-model.pt")
    f_optimizer = os.path.join(out_dir, f"{prefix}-optimizer.pt")
    f_scaler = os.path.join(out_dir, f"{prefix}-scaler.pt")
    f_state = os.path.join(out_dir, f"{prefix}-state.json")

    msg = dict(
        event="Saving checkpoint...",
        model=f_model,
        optimizer=f_optimizer,
        scaler=f_scaler,
        state=f_state,
    )

    files = [f_model, f_optimizer]
    if scaler:
        files.append(f_scaler)

    if state:
        assert callable(state.to_json) and callable(state.from_json)
        files.append(f_state)

    if uploading_event.is_set():
        logger.warn("Still uploading previous checkpoint, will not save this one locally or to S3")
        return

    if dry_run:
        msg["event"] += " (--dry-run)"
        logger.info(msg)
    else:
        logger.info(msg)
        # Prevent corrupted checkpoints if terminated during torch.save

        if permanent:
            with open(os.path.join(out_dir, f"{prefix}-config.json"), "w") as f:
                logger.debug(f"Saving config to: {f.name}")
                json.dump(config, f, indent=4, sort_keys=False)

        if optimize_local_storage:
            # Use "...~" as a lockfile
            # While the lockfile exists, the original file is corrupted
            # (i.e. save() was interrupted => S3 download is needed to load())

            # NOTE: bulk create and remove lockfiles to prevent mixing up
            #       different checkpoints when only 1 or 2 files get saved

            pathlib.Path(f"{f_model}~").touch()
            pathlib.Path(f"{f_optimizer}~").touch()
            if scaler:
                pathlib.Path(f"{f_scaler}~").touch()
            if state:
                pathlib.Path(f"{f_state}~").touch()

            torch.save(model.state_dict(), f_model)
            torch.save(optimizer.state_dict(), f_optimizer)
            if scaler:
                torch.save(scaler.state_dict(), f_scaler)
            if state:
                with open(os.path.join(out_dir, f_state), "w") as f:
                    f.write(state.to_json())

            os.unlink(f"{f_model}~")
            os.unlink(f"{f_optimizer}~")
            if scaler:
                os.unlink(f"{f_scaler}~")
            if state:
                os.unlink(f"{f_state}~")
        else:
            # Use temporary files to ensure the original one is always good
            # even if the .save is interrupted
            # NOTE: first save all, then move all, to prevent mixing up
            #       different checkpoints when only 1 or 2 files get saved
            torch.save(model.state_dict(), f"{f_model}.tmp")
            torch.save(optimizer.state_dict(), f"{f_optimizer}.tmp")
            if scaler:
                torch.save(scaler.state_dict(), f"{f_scaler}.tmp")
            if state:
                with open(os.path.join(out_dir, f"{f_state}.tmp"), "w") as f:
                    f.write(state.to_json())

            shutil.move(f"{f_model}.tmp", f_model)
            shutil.move(f"{f_optimizer}.tmp", f_optimizer)
            if scaler:
                shutil.move(f"{f_scaler}.tmp", f_scaler)
            if state:
                shutil.move(f"{f_state}.tmp", f_state)

    if not s3_config:
        return

    if uploading_event.is_set():
        logger.warn("Still uploading previous checkpoint, will not upload this one to S3")
        return

    uploading_event.set()
    logger.debug("uploading_event: set")

    bucket = s3_config["bucket_name"]
    s3_dir = s3_config["s3_dir"]
    s3 = init_s3_client()

    files.insert(0, os.path.join(out_dir, f"{prefix}-config.json"))

    try:
        for f in files:
            key = f"{s3_dir}/{os.path.basename(f)}"
            msg = f"Uploading to s3://{bucket}/{key} ..."

            if dry_run:
                logger.info(f"{msg} (--dry-run)")
            else:
                logger.info(msg)
                size_mb = os.path.getsize(f) / 1e6

                if size_mb < 100:
                    logger.debug("Uploading as single chunk")
                    s3.upload_file(f, bucket, key)
                elif size_mb < 1000:  # 1GB
                    logger.debug("Uploding on chunks of 50MB")
                    tc = TransferConfig(multipart_threshold=50 * 1024 * 1024, use_threads=True)
                    s3.upload_file(f, bucket, key, Config=tc)
                else:
                    logger.debug("Uploding on chunks of 500MB")
                    tc = TransferConfig(multipart_threshold=500 * 1024 * 1024, use_threads=True)
                    s3.upload_file(f, bucket, key, Config=tc)

                logger.info(f"Uploaded: s3://{bucket}/{key}")

    finally:
        uploading_event.clear()
        logger.debug("uploading_event: cleared")


def save_buffer_async(
    run_id,
    logger,
    dry_run,
    buffer,
    env_config,
    s3_config,
    allow_skip,
    uploading_cond,
    uploading_event_buf,
    optimize_local_storage
):
    # If a previous upload is still in progress, block here until it finishes
    logger.debug("Trying to obtain lock (main thread)...")
    with uploading_cond:
        logger.debug("Obtained lock (main thread); starting sub-thread...")

        thread = threading.Thread(target=_save_buffer, kwargs=dict(
            logger=logger,
            dry_run=dry_run,
            buffer=buffer,
            # out_dir=config["run"]["out_dir"],
            run_id=run_id,
            env_config=env_config,
            s3_config=s3_config,
            uploading_cond=uploading_cond,
            uploading_event=uploading_event_buf,
            optimize_local_storage=optimize_local_storage,
            allow_skip=allow_skip,
        ))
        thread.start()
        # sub-thread should save the buffer to temp dir and notify us
        logger.debug("Waiting on cond (main thread) ...")
        if not uploading_cond.wait(timeout=10):
            logger.error("Thread for buffer upload did not start properly")
        logger.debug("Notified; releasing lock (main thread) ...")
        uploading_cond.notify_all()


# NOTE: this assumes no old observations are left in the buffer
def _save_buffer(
    logger,
    dry_run,
    buffer,
    run_id,
    env_config,
    s3_config,
    uploading_cond,
    uploading_event,
    optimize_local_storage,
    allow_skip=True
):
    # XXX: this is a sub-thread
    # Parent thread has released waits for us to notify via the cond that we have
    # saved the buffer to files, so it can start filling the buffer with new
    # while we are uploading.
    # However, it won't be able to start a new upload until this one finishes.

    # XXX: Saving to tempdir (+deleting afterwards) to prevent disk space issues
    # bufdir = os.path.join(out_dir, "samples", "%s-%d" % (run_id, time.time()))
    # msg = f"Saving buffer to {bufdir}"
    # if dry_run:
    #     logger.info(f"{msg} (--dry-run)")
    # else:
    #     logger.info(msg)

    cache_dir = s3_config["cache_dir"]
    s3_dir = s3_config["s3_dir"]
    bucket = s3_config["bucket_name"]

    # No need to store temp files if we can bail early
    if allow_skip and uploading_event.is_set():
        logger.warn("Still uploading previous buffer, will not upload this one to S3")
        # We must still unblock the main thread
        with uploading_cond:
            logger.debug("Obtained lock (sub-thread); notify_all() ...")
            uploading_cond.notify_all()
        return

    now = time.time_ns() / 1000
    fname = f"transitions-{buffer.containers['obs'].shape[0]}-{now:.0f}.npz"
    s3_path = f"{s3_dir}/{fname}"
    local_path = f"{cache_dir}/{s3_path}"
    msg = f"Saving buffer to {local_path}"
    to_save = {k: v.cpu().numpy() for k, v in buffer.containers.items()}
    to_save["md"] = {"env_config": env_config, "s3_config": s3_config}

    if dry_run:
        logger.info(f"{msg} (--dry-run)")
    else:
        logger.info(msg)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        np.savez_compressed(local_path, **to_save)

    def do_upload():
        s3 = init_s3_client()
        msg = f"Uploading to s3://{bucket}/{s3_path} ..."

        if dry_run:
            logger.info(f"{msg} (--dry-run + sleep(10))")
            time.sleep(10)
        else:
            logger.info(msg)
            s3.upload_file(local_path, bucket, s3_path)

        logger.info(f"Uploaded: s3://{bucket}/{s3_path}")

        if optimize_local_storage and os.path.exists(local_path):
            logger.info(f"Remove {local_path}")
            os.unlink(local_path)

    # Buffer saved to local disk =>
    # Notify parent thread so it can now proceed with collecting new obs in it
    # XXX: this must happen AFTER the buffer is fully dumped to local disk
    logger.debug("Trying to obtain lock for notify (sub-thread)...")
    with uploading_cond:
        logger.debug("Obtained lock (sub-thread); notify_all() ...")
        uploading_cond.notify_all()

    if allow_skip:
        # We will simply skip the upload if another one is still in progress
        # (useful if training while also collecting samples)
        if uploading_event.is_set():
            logger.warn("Still uploading previous buffer, will not upload this one to S3")
            return
        uploading_event.set()
        logger.debug("uploading_event: set")
        do_upload()
        uploading_event.clear()
        logger.debug("uploading_event: cleared")
    else:
        # We will hold the cond lock until we are done with the upload
        # so parent will have to wait before starting us again
        # (useful if collecting samples only)
        logger.debug("Trying to obtain lock for upload (sub-thread)...")
        with uploading_cond:
            logger.debug("Obtained lock; Proceeding with upload (sub-thread) ...")
            do_upload()
            logger.info("Successfully uploaded buffer to s3; releasing lock ...")
