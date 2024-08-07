import os
import sys
import signal
import argparse
import importlib
import datetime


def handle_signal(signum, frame):
    print("*** [main.py] received signal %s ***" % signum)
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", metavar="<strategy>", default="pbt", help="strategy (pbt, pb2)")
    parser.add_argument("-a", metavar="<algo>", default="mppo", help="rl algo module (mppo, mppo_dna, mppg, mqrdqn, ...)")
    parser.add_argument("-n", metavar="<name>", default="PBT-{datetime}", help="experiment name")
    parser.add_argument("-R", metavar="<path>", help="resume experiment from a single saved agent")
    parser.add_argument('-o', metavar="<cfgpath>=<value>", action='append', help='Override config value based on dot-delimited path')
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.epilog = """

Available strategies:
  pbt           (default) Population-based training (PBT)
  pb2           Population-based bandits (PB2)

Available algos:
  mppo          Maskable Proximal Policy Optimization (MPPO)
  mppo_dna      MPPO with Dual Network Arch (MPPO-DNA)
  mppg          Maskable Phasic Policy Gradient (PPG)
  mqrdqn        Maskable Quantile-Regression DQN (QRDQN)
  ...           (refer to the contents of the rl/algos directory)

Examples:
  python -m rl.raytune.main -n "PBT-experiment1-{datetime}"
  python -m rl.raytune.main -R "/path/to/saved/agent.pt" -o "_raytune.hyperparam_mutations.num_steps=[32, 64, 128, 256]"
  python -m rl.raytune.main -s pb2 -a mppg -n "..."
"""
    # XXX: env vars must be set *before* importing ray/wandb modules

    os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
    os.environ["RAY_DEDUP_LOGS"] = "0"
    os.environ["WANDB_SILENT"] = "true"

    args = parser.parse_args()
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        # XXX: can't use relative imports here
        mod = importlib.import_module(f"rl.raytune.{args.s}")
    except ModuleNotFoundError as e:
        if e.name == args.s:
            print("Unknown strategy: %s" % args.s)
            sys.exit(1)
        raise

    experiment_name = args.n.format(datetime=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    mod.main(args.a, experiment_name, args.R, args.o or [])
