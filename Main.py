import argparse
import TrainModel
import os
import ast
import torch


#Training Code

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=ast.literal_eval, default=True)
    parser.add_argument("--use_cuda", type=ast.literal_eval, default=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--eval_save", type=ast.literal_eval, default=True,
                        help="Whether to save results during evaluation")
    parser.add_argument("--eval_per_img", type=ast.literal_eval, default=False,
                        help="Whether to save metrics per img during evaluation")
    parser.add_argument("--trainset", type=str, default=r"/data/liao/seqMSEC/INPUT_IMAGES/")
    parser.add_argument("--testset", type=str, default=r"/data/liao/seqMSEC/INPUT_IMAGES/")
    parser.add_argument("--testGTset", type=str, default=r"/data/liao/seqMSEC/GT_IMAGES/expert_c_testing_set/")
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument('--ckpt', default=None, type=str, help='name of the checkpoint to load')
    parser.add_argument('--experiment_path', default='./experiments/', type=str,
                        metavar='PATH', help='path to save experiment data')
    parser.add_argument('--experiment_name', default='experimentLatest', type=str,
                        metavar='PATH', help='experiment name')
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--decay_interval", type=int, default=1000)
    parser.add_argument("--decay_ratio", type=float, default=0.1)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--epochs_per_eval", type=int, default=1)  #
    parser.add_argument("--epochs_per_save", type=int, default=1000)  #
    # parser.add_argument("--epochs_warmup_sec", type=int, default=10)
    parser.add_argument("--epochs_stable", type=int, default=30)  #
    parser.add_argument("--epochs_warmup", type=int, default=10)  #

    return parser.parse_args()


def main(cfg):
    cfg.fused_img_path = os.path.join(cfg.experiment_path, cfg.experiment_name, 'fused_img_path')
    cfg.weight_map_path = os.path.join(cfg.experiment_path, cfg.experiment_name, 'weight_map_path')
    cfg.corrected_img_path = os.path.join(cfg.experiment_path, cfg.experiment_name, 'corrected_img_path')
    cfg.ckpt_path = os.path.join(cfg.experiment_path, cfg.experiment_name, 'ckpt_path')

    # Create directories if they don't exist
    os.makedirs(cfg.fused_img_path, exist_ok=True)
    os.makedirs(cfg.weight_map_path, exist_ok=True)
    os.makedirs(cfg.corrected_img_path, exist_ok=True)
    os.makedirs(cfg.ckpt_path, exist_ok=True)
    torch.cuda.set_device(cfg.device)
    t = TrainModel.Trainer(cfg)
    if cfg.train:
        t.fit()
    else:
        t.eval(0)


if __name__ == "__main__":
    config = parse_config()

    main(config)
