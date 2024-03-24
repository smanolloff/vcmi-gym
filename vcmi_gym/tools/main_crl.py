# =============================================================================
# Copyright 2024 Simeon Manolov <s.manolloff@gmail.com>.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import os
import yaml
import re
import argparse
import signal
import sys
import importlib

from . import common

# NOTE (MacOS ONLY):
# To prevent annoying ApplePersistenceIgnoreState message:
# $ defaults write org.python.python ApplePersistenceIgnoreState NO


def handle_signal(signum, frame):
    print("*** [main.py] received signal %s ***" % signum)
    sys.exit(0)


def run(action, cfg, group_id, run_id, resume, cfgpath):
    try:
        # XXX: can't use relative imports here
        mod = importlib.import_module(f"vcmi_gym.tools.crl.{action}")
    except ModuleNotFoundError:
        print("Unknown action: %s" % action)
        sys.exit(1)

    cfg["cfg_file"] = cfgpath

    os.environ["WANDB_SILENT"] = "true"

    if group_id is not None:
        cfg["group_id"] = group_id

    if run_id is not None:
        cfg["run_id"] = run_id
    elif not cfg.get("run_id", None):
        cfg["run_id"] = common.gen_id()

    if resume is not None:
        cfg["resume"] = resume

    assert re.match(r"^[A-Za-z0-9][A-Za-z0-9_-]+[A-Za-z0-9]$", cfg["group_id"]), \
        "invalid group_id: %s" % cfg["group_id"]

    args = mod.Args(**cfg)
    print("Starting run %s with seed %s" % (args.run_id, args.seed))

    run_duration, run_values = common.measure(mod.main, dict(args=args))
    common.save_run_metadata(
        action=action,
        cfg=vars(args),
        duration=run_duration,
        values=dict(run_values),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", help=argparse.SUPPRESS)
    parser.add_argument("-g", metavar="GROUP_ID", help="group_id")
    parser.add_argument("-r", metavar="RUN_ID", help="run_id")
    parser.add_argument("-R", help="resume training", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "-c",
        metavar="FILE",
        type=argparse.FileType("r"),
        help="config file, defaults to config/<action>.yml",
    )
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.usage = "%(prog)s [options] <action> [<value>]"
    parser.epilog = """
action:
  mppo              train using Maskable Proximal Policy Optimization (MPPO)
  mppo_heads        train using MPPO with a multi-head policy
  help              print this help message

examples:
  %(prog)s -c path/to/config.yml mppo
"""

    args = parser.parse_args()

    signal.signal(signal.SIGTERM, handle_signal)
    cfg = {}

    if args.c is not None:
        print("Loading configuration from %s" % args.c.name)
        cfg = yaml.safe_load(args.c)
        args.c.close()

    run(args.action, cfg, args.g, args.r, args.R, cfgpath=getattr(args.c, "name", None))


if __name__ == "__main__":
    main()
