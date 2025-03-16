import boto3
import os
import glob
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


def file_exists_in_s3(s3_client, bucket_name, s3_key):
    """ Checks if a file exists in the S3 bucket """
    try:
        s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        return True  # File exists
    except s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False  # File does not exist
        else:
            raise  # Some other error


def upload_files_to_s3(localdir, bucket_name, s3_prefix, aws_access_key, aws_secret_key, region_name):
    assert s3_prefix

    logger = StructuredLogger(filename=os.path.join("s3uploader.log"))

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=region_name
    )

    request = {"Bucket": bucket_name, "Prefix": s3_prefix}
    s3_keys = []
    while True:
        response = s3_client.list_objects_v2(**request)
        s3_keys.extend([obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.npz')])
        if "NextContinuationToken" in response:
            request["ContinuationToken"] = response["NextContinuationToken"]
        else:
            break

    npz_files = glob.glob(os.path.join(localdir, "*", "*.npz"), recursive=True)
    json_files = glob.glob(os.path.join(localdir, "*", "metadata.json"), recursive=True)

    logger.log("Found %d files" % (len(npz_files) + len(json_files)))
    pattern = re.compile(fr"{localdir}/(.*-.*)/(.*)\.(npz|json)")

    for local_path in npz_files + json_files:
        m = pattern.match(local_path)
        suffix = m[1]
        t = m[2]
        ext = m[3]
        s3_key = f"{s3_prefix}/{t}-{suffix}.{ext}"

        # XXX: TMP linkinkg
        target = "../../../" + local_path
        link_name = os.path.dirname(__file__) + "/.cache/" + os.path.basename(s3_key)
        if not os.path.exists(link_name):
            logger.log(f"Link {local_path} to s3://{bucket_name}/{s3_key}...")
            os.symlink(target, link_name)
        continue
        # EOF: TMP linkinkg

        if s3_key in s3_keys:
            logger.log(f"Skip {s3_key} (already exists in S3)")
        else:
            logger.log(f"Upload {local_path} to s3://{bucket_name}/{s3_key}...")
            s3_client.upload_file(local_path, bucket_name, s3_key)

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
# Files will be uploaded to s3 as {s3_prefix}/{type}-{dir}, e.g.:
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

upload_files_to_s3(
    localdir="data/autoencoder/samples/v8",
    bucket_name="vcmi-gym",  # see big note above
    s3_prefix="v8",
    aws_access_key=os.environ["S3_RW_ACCESS_KEY"],
    aws_secret_key=os.environ["S3_RW_SECRET_KEY"],
    region_name="eu-north-1"
)
