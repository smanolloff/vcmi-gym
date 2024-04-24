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
import string
import random
import importlib

# NOTE (MacOS ONLY):
# To prevent annoying ApplePersistenceIgnoreState message:
# $ defaults write org.python.python ApplePersistenceIgnoreState NO


def handle_signal(signum, frame):
    print("*** [main.py] received signal %s ***" % signum)
    sys.exit(0)


def gen_id():
    population = string.ascii_lowercase + string.digits
    return str.join("", random.choices(population, k=8))


def run(action, cfg, group_id, run_id, resume, cfgpath):
    try:
        # XXX: can't use relative imports here
        mod = importlib.import_module(f"rl.algos.{action}.{action}")
    except ModuleNotFoundError as e:
        if e.name == action:
            print("Unknown action: %s" % action)
            sys.exit(1)
        raise

    cfg["cfg_file"] = cfgpath

    os.environ["WANDB_SILENT"] = "true"

    if os.environ.get("NO_WANDB", "false") == "true":
        cfg["wandb_project"] = None

    if group_id is not None:
        cfg["group_id"] = group_id

    if run_id is not None:
        cfg["run_id"] = run_id
    elif not cfg.get("run_id", None):
        cfg["run_id"] = gen_id()

    if resume is not None:
        cfg["resume"] = resume

    assert re.match(r"^[A-Za-z0-9][A-Za-z0-9_-]+[A-Za-z0-9]$", cfg["group_id"]), \
        "invalid group_id: %s" % cfg["group_id"]

    args = mod.Args(**cfg)
    print("Starting run %s with seed %s" % (args.run_id, args.seed))

    mod.main(args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", help=argparse.SUPPRESS)
    parser.add_argument("-g", metavar="GROUP_ID", help="group_id")
    parser.add_argument("-r", metavar="RUN_ID", help="run_id")
    parser.add_argument("-R", help="resume training", action='store_true')
    parser.add_argument("-f", metavar="FILE", help="config file")

    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.usage = "%(prog)s [options] <action> [<value>]"
    parser.epilog = """
action:
  mppo              train using Maskable Proximal Policy Optimization (MPPO)
  qrdqn             train using Maskable Quantile-Regression DQN (QRDQN)
  help              print this help message

examples:
  %(prog)s -f path/to/config.yml mppo
  %(prog)s -g mygroup -r myrun -R mppo
"""

    args = parser.parse_args()

    signal.signal(signal.SIGTERM, handle_signal)
    cfg = {}

    if args.f is not None:
        print("Loading configuration from %s" % args.f)
        with open(args.f, "rb") as f:
            cfg = yaml.safe_load(f)

    run(args.action, cfg, args.g, args.r, args.R, cfgpath=args.f)


if __name__ == "__main__":
    # Run from vcmi-gym root:
    # $ python -m rl.algos.main -f rl/algos/mppo/mppo-config.yml mppo
    main()
