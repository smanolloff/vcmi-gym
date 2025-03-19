import boto3
import os
import sys
from datetime import datetime
import re
import yaml
import json


class StructuredLogger:
    def __init__(self, filename):
        self.filename = filename
        self.log(dict(filename=filename))

    def log(self, obj):
        timestamp = datetime.utcnow().isoformat(timespec='milliseconds')
        if isinstance(obj, dict):
            log_obj = dict(timestamp=timestamp, message=obj)
        else:
            log_obj = dict(timestamp=timestamp, message=dict(string=obj))

        print(yaml.dump(log_obj, sort_keys=False))
        with open(self.filename, "a+") as f:
            f.write(json.dumps(log_obj) + "\n")


def download_files_from_s3(localdir, bucket_name, s3_dir, aws_access_key, aws_secret_key, region_name):
    assert s3_dir

    logger = StructuredLogger(filename=os.path.join("s3downloader.log"))

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=region_name
    )

    request = {"Bucket": bucket_name, "Prefix": s3_dir}
    s3_keys = []
    while True:
        response = s3_client.list_objects_v2(**request)
        s3_keys.extend([obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.npz') or obj['Key'].endswith('.json')])
        if "NextContinuationToken" in response:
            request["ContinuationToken"] = response["NextContinuationToken"]
        else:
            break

    logger.log("Found %d S3 keys" % len(s3_keys))
    pattern = re.compile(fr"{s3_dir}/(.*?)-(.*)\.(npz|json)")

    for s3_key in s3_keys:
        m = pattern.match(s3_key)
        t = m[1]
        suffix = m[2]
        ext = m[3]
        # local_path = f"{localdir}/{suffix}/{t}.{ext}"
        local_path = f"{localdir}/{t}-{suffix}.{ext}"

        if os.path.exists(local_path):
            logger.log(f"Skip {s3_key} (found {local_path})")
        else:
            logger.log(f"Download s3://{bucket_name}/{s3_key} to {local_path}...")
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3_client.download_file(bucket_name, s3_key, local_path)

    logger.log("Done.")


# `localdir` should contain subdirectories, e.g:
#
#   ./eqmnojqh-00000/
#   ./eqmnojqh-00001/
#   ./eqmnojqh-00002/
#   ./eqmnojqh-.../
#   ./tfklvbdl-00000/
#   ./tfklvbdl-00001/
#   ./tfklvbdl-00002/
#   ./tfklvbdl-.../
#   ./foo/
#
# Each subdirectory should contain {type}.{ext} npz files and a metadata.json file:
#   ./action.npz
#   ./metadata.json
#   ./done.npz
#   ./obs.npz
#   ./reward.npz
#   ./mask.npz
#   ./bar.npz
#
# Files will be uploaded to s3 as {s3_dir}/{type}-{dir}, e.g.:
#
#   v8/action-eqmnojqh-00000.npz
#   v8/action-tfklvbdl-00000.npz
#   v8/action-foo.npz
#   ...
#   v8/bar-eqmnojqh-00000.npz
#   v8/bar-tfklvbdl-00000.npz
#   v8/bar-foo.npz
#   ...
#

download_files_from_s3(
    localdir=sys.argv[1],
    bucket_name="vcmi-gym",  # see big note above
    # s3_dir="v8-100k",  # don't use -- cause OOM when loaded in parallel
    s3_dir="v8",
    aws_access_key=os.environ["AWS_ACCESS_KEY"],
    aws_secret_key=os.environ["AWS_SECRET_KEY"],
    region_name="eu-north-1"
)
