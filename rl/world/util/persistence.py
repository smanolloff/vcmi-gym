import os
import pathlib
import shutil
import torch
import numpy as np
import time
import threading
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


def load_local_or_s3_checkpoint(
    logger,
    dry_run,
    checkpoint_s3_config,
    optimize_local_storage,
    device,
    out_dir,
    run_id,
    what,
    torch_obj,
    **load_kwargs
):
    filename = "%s/%s-%s.pt" % (out_dir, run_id, what)
    logger.info(f"Load {what} from {filename}")

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
    if checkpoint_s3_config and not os.path.exists(filename):
        logger.debug("Local file does not exist, try S3")

        s3_filename = f"{checkpoint_s3_config['s3_dir']}/{os.path.basename(filename)}"
        logger.info(f"Download s3://{checkpoint_s3_config['bucket_name']}/{s3_filename} ...")

        if os.path.exists(f"{filename}.tmp"):
            os.unlink(f"{filename}.tmp")
        init_s3_client().download_file(checkpoint_s3_config["bucket_name"], s3_filename, f"{filename}.tmp")
        shutil.move(f"{filename}.tmp", filename)
    torch_obj.load_state_dict(torch.load(filename, weights_only=True, map_location=device), **load_kwargs)

    if not dry_run and not optimize_local_storage:
        backname = "%s-%d.pt" % (filename.removesuffix(".pt"), time.time())
        logger.debug(f"Backup resumed model weights as {backname}")
        shutil.copy2(filename, backname)


def save_checkpoint(logger, dry_run, model, optimizer, scaler, out_dir, run_id, optimize_local_storage, s3_config, uploading_event):
    f_model = os.path.join(out_dir, f"{run_id}-model.pt")
    f_optimizer = os.path.join(out_dir, f"{run_id}-optimizer.pt")
    f_scaler = os.path.join(out_dir, f"{run_id}-scaler.pt")
    msg = dict(
        event="Saving checkpoint...",
        model=f_model,
        optimizer=f_optimizer,
        scaler=f_scaler,
    )

    files = [f_model, f_optimizer]
    if scaler:
        files.append(f_scaler)

    if uploading_event.is_set():
        logger.warn("Still uploading previous checkpoint, will not save this one locally or to S3")
        return

    if dry_run:
        msg["event"] += " (--dry-run)"
        logger.info(msg)
    else:
        logger.info(msg)
        # Prevent corrupted checkpoints if terminated during torch.save

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

            torch.save(model.state_dict(), f_model)
            torch.save(optimizer.state_dict(), f_optimizer)
            if scaler:
                torch.save(scaler.state_dict(), f_scaler)

            os.unlink(f"{f_model}~")
            os.unlink(f"{f_optimizer}~")
            if scaler:
                os.unlink(f"{f_scaler}~")
        else:
            # Use temporary files to ensure the original one is always good
            # even if the .save is interrupted
            # NOTE: first save all, then move all, to prevent mixing up
            #       different checkpoints when only 1 or 2 files get saved
            torch.save(model.state_dict(), f"{f_model}.tmp")
            torch.save(optimizer.state_dict(), f"{f_optimizer}.tmp")
            if scaler:
                torch.save(scaler.state_dict(), f"{f_scaler}.tmp")

            shutil.move(f"{f_model}.tmp", f_model)
            shutil.move(f"{f_optimizer}.tmp", f_optimizer)
            if scaler:
                shutil.move(f"{f_scaler}.tmp", f_scaler)

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

    files.insert(0, os.path.join(out_dir, f"{run_id}-config.json"))

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
