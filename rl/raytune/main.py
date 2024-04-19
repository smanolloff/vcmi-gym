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
    parser.add_argument("-s", "--strategy", metavar="<strategy>", default="pbt", help="strategy (pbt, pb2)")
    parser.add_argument("-a", "--algo", metavar="<algo>", default="mppo", help="rl algo module (mppo, mppo_dna, mppg, mqrdqn, ...)")
    parser.add_argument("-n", metavar="<name>", default="PBT-{datetime}", help="experiment name")
    parser.add_argument("-R", metavar="<path>", help="resume experiment from path")
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    parser.usage = "%(prog)s [options] <algo> <experiment_name>"
    parser.epilog = """

Available strategies:
  pbt           (default) Population-based training (PBT)
  pb2           Population-based bandits (PB2)

Available algos:
  mppo          Maskable Proximal Policy Optimization (MPPO)
  mppo_dna      MPPO with Dual Network Arch (MPPO-DNA)
  mppg          Maskable Phasic Policy Gradient (PPG)
  mqrdqn        Maskable Quantile-Regression DQN (QRDQN)

Examples:
  %(prog)s "PBT-experiment1-{datetime}"
  %(prog)s -R "/path/to/PBT-experiment1-20240414_141602"
  %(prog)s -s pb2 -a mppg "..."
"""
    # XXX: env vars must be set *before* importing ray/wandb modules

    # this makes the "storage" arg redundant. By default, TUNE_RESULT_DIR
    # is $HOME/ray_results and "storage" just *copies* everything into data
    os.environ["TUNE_RESULT_DIR"] = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
    os.environ["RAY_DEDUP_LOGS"] = "0"
    os.environ["WANDB_SILENT"] = "true"

    args = parser.parse_args()
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        # XXX: can't use relative imports here
        mod = importlib.import_module(f"rl.raytune.{args.strategy}")
    except ModuleNotFoundError as e:
        if e.name == args.strategy:
            print("Unknown strategy: %s" % args.strategy)
            sys.exit(1)
        raise

    experiment_name = args.n.format(datetime=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    mod.main("mppo", experiment_name, args.R)
