"""
Script to start slurm training for `accelerate launch`.

This script will generate a **sbatch** file, and call the file via `sbatch`.
The sbatch command **DO NOT** support interactive debug (e.g. pdb), and will
write the **stdout** and **stderr** to `log-out` path.

Usage:

```bash
# dry run
python submit.py --job-name powerpaint --gpus 16 --dry-run \
    train_ppt1_sd15.py --config configs/ppt1_sd15.yaml

# or direct start!
python submit.py --job-name powerpaint --gpus 16 \
    train_ppt1_sd15.py --config configs/ppt1_sd15.yaml
```
"""

import os
from argparse import ArgumentParser
from datetime import datetime


parser = ArgumentParser()
parser.add_argument("--job-name", default="powerpaint")
parser.add_argument(
    "--gpus",
    type=int,
    default=8,
    help="**Total** gpu you want to run your command.",
)
parser.add_argument(
    "--gpus-per-nodes",
    type=int,
    default=8,
    help="number of nodes",
)
parser.add_argument(
    "--cpus-per-node",
    type=int,
    default=128,
    help="cpus for each **node**.",
)

parser.add_argument(
    "--log-path",
    type=str,
    default="runs",
    help=("path of the log files (stdout, stderr). " "If not passed, will be `runs/JOB_NAME_MMDD_HHMM.sh`"),
)
parser.add_argument(
    "--script-path",
    help=("the name of the sbatch script. " "If not passed, will be `JOB_NAME_MMDD_HHMM.sh`"),
)

parser.add_argument(
    "--dry-run",
    action="store_true",
    help="If true, will generate the script but do not run.",
)

parser.add_argument(
    "-x",
    nargs="+",
    type=str,
    help="exclude machine",
)

# args = parser.parse_args()
args, cmd_list = parser.parse_known_args()

print(args)
print(cmd_list)


def main():
    gpus = args.gpus
    gpus_per_nodes = args.gpus_per_nodes
    cpus_per_node = args.cpus_per_node

    assert (
        gpus_per_nodes <= 8 and gpus_per_nodes >= 1
    ), f"gpus_per_node must be in [1, 8], but receive {gpus_per_nodes}."

    if gpus <= gpus_per_nodes:
        n_node = 1
        gpus_per_nodes = gpus
    else:
        assert gpus % gpus_per_nodes == 0, "gpus must be divided by gpus_per_nodes."
        n_node = gpus // gpus_per_nodes

    MMDD_HHMM = datetime.now().strftime("%m%d_%H%M")
    if args.log_path is None:
        log_path = f"runs/{args.job_name}_{MMDD_HHMM}"
    else:
        log_path = args.log_path
    os.makedirs(log_path, exist_ok=True)

    # start write script
    if args.script_path is None:
        script_path = f"runs/{args.job_name}_{MMDD_HHMM}.batchscript"
    else:
        script_path = args.script_path

    with open(script_path, "w") as file:
        header = (
            "#!/bin/bash\n"
            f"#SBATCH --job-name={args.job_name}\n"
            "#SBATCH -p mm_lol\n"
            f"#SBATCH --output={log_path}/O-%x.%j\n"
            f"#SBATCH --error={log_path}/E-%x.%j\n"
            f"#SBATCH --nodes={n_node}                # number of nodes\n"
            "#SBATCH --ntasks-per-node=1              # number of MP tasks\n"
            f"#SBATCH --gres=gpu:{gpus_per_nodes}     # number of GPUs per node\n"
            f"#SBATCH --cpus-per-task={cpus_per_node} # number of cores per tasks\n"
        )

        network = (
            "######################\n"
            "#### Set network #####\n"
            "######################\n"
            "head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)\n"
            "export MASTER_PORT=$((12000 + $RANDOM % 20000))\n"
            "######################\n"
        )

        cmd_string = " ".join(cmd_list)
        print(args.x)
        if args.x is not None:
            srun_string = f"srun -x {' '.join(list(args.x))} "
        else:
            srun_string = "srun "
        launcher = (
            f"{srun_string} "
            "accelerate launch --multi_gpu "
            f"--num_processes {gpus} "
            "--num_machines ${SLURM_NNODES} "
            "--machine_rank ${SLURM_NODEID} "
            "--rdzv_backend c10d "
            "--main_process_ip $head_node_ip "
            f"--main_process_port ${{MASTER_PORT}} "
        )
        launcher += cmd_string

        file.write(header)
        file.write("\n")
        file.write(network)
        file.write("\n")
        file.write(launcher)
        file.write("\n")

    print(f"Write script to {script_path}.")

    if not args.dry_run:
        os.system(f"sbatch {script_path}")
        return

    print(f"You can run the script manually via 'sbatch {script_path}'")


if __name__ == "__main__":
    main()
