import argparse
import subprocess


parser = argparse.ArgumentParser(description="run")
parser.add_argument("--device", type=str)
parser.add_argument("--run", type=int)
parser.add_argument("--ckpt", type=str)
args = parser.parse_args()

if args.run == 0:
    # training
    subprocess.call(f"bash experiments/conformer_prediction/rebind.sh {args.device}", shell=True)

if args.run == 1:
    # evaluation
    ckpt = args.ckpt
    subprocess.call(f"export CUDA_VISIBLE_DEVICES={args.device} && python -m evaluate --data_dir datasets/ --dataset QM9 --mode random --split test --log_file logs/conformer_prediction/QM9/REBIND_result.txt --REBIND_checkpoint {ckpt} --device cuda:{args.device} --removeHs --log_results 1", shell=True)

