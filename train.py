import argparse
import os
import time
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from utils.dataset import get_loader
from model import myModel
from datetime import datetime
from utils.metrics import Evaluator
from utils.loss_funcs import (
    EdgeAwareLoss,
    SSIMLoss,
    L1_Charbonnier_loss,
    PerceptualLoss,
)
from utils.CIDNet import CIDNet


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class Trainer(object):
    def __init__(self, args):
        """
        Initialize the Trainer with arguments from the command line or defaults.
        """
        self.args = args
        self.evaluator = Evaluator()

        self.deep_model = myModel(
            in_channels=3, feature_channels=32, use_white_balance=True
        )
        self.deep_model = self.deep_model.to("cuda")

        self.hvi_net = CIDNet().cuda()
        pth = r"utils/CIDNet_weight_LOLv2_bestSSIM.pth"
        self.hvi_net.load_state_dict(torch.load(pth, map_location="cuda"))
        self.hvi_net.eval()

        # Create a directory to save model weights, organized by timestamp.
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_save_path = os.path.join(
            args.save_path, args.model_name, args.dataset, now_str
        )

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)

        self.optim = optim.AdamW(
            self.deep_model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
        )
        if args.scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optim, args.epoch, eta_min=args.lr * 1e-4
            )
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optim, step_size=args.decay_epoch, gamma=args.decay_rate
            )
        #### ALL DATASET RESIZED TO 256x256 ####
        if args.dataset == "EUVP-d":  # 256x256
            args.train_root = "/ssd6/UnderWaterDataset/train/EUVP_d/"
            args.val_root = "/ssd6/UnderWaterDataset/val/EUVP_d/"
            args.datasize = 256
            args.resize = False
        elif args.dataset == "EUVP-s":  # 320x240
            args.train_root = "/ssd6/UnderWaterDataset/train/EUVP_s/"
            args.val_root = "/ssd6/UnderWaterDataset/val/EUVP_s/"
            args.datasize = 256  # while validating, don't resize
            args.resize = True
        elif args.dataset == "UIEB":  # 1280x~800 => 256x256
            args.train_root = "/ssd6/UnderWaterDataset/train/UIEB/"
            args.val_root = "/ssd6/UnderWaterDataset/val/UIEB/"
            args.datasize = 256
            args.resize = True
        elif args.dataset == "UFO":  # 320x240
            args.train_root = "/ssd6/UnderWaterDataset/train/UFO_120/"
            args.val_root = "/ssd6/UnderWaterDataset/val/UFO_120/"
            args.datasize = 256  # while validating, don't resize
            args.resize = True
        elif args.dataset == "LSUI":  # 720x405 => 256x256
            args.train_root = "/ssd6/UnderWaterDataset/train/LSUI/"
            args.val_root = "/ssd6/UnderWaterDataset/val/LSUI/"
            args.datasize = 256
            args.resize = True

        ## loss functions
        self.vggL = PerceptualLoss()
        self.L1L = L1_Charbonnier_loss()
        self.ssimL = SSIMLoss(device="cuda", window_size=5)
        self.edgeL = EdgeAwareLoss(loss_type="l2", device="cuda")

    def training(self):
        """
        Main training loop that iterates over the training dataset for several steps (max_iters).
        Prints intermediate losses and evaluates on holdout dataset periodically.
        """
        best_psnr = 0.0
        best_round = []
        torch.cuda.empty_cache()
        train_data_loader = get_loader(
            self.args.train_root,
            self.args.train_batch_size,
            self.args.datasize,
            train=True,
            resize=self.args.resize,
            num_workers=self.args.num_workers,
            shuffle=True,
            pin_memory=True,
        )
        self.deep_model.train()
        for epoch in range(1, self.args.epoch + 1):
            loop = tqdm(
                enumerate(train_data_loader), total=len(train_data_loader), leave=False
            )
            loss_mean = 0.0
            for _, (x, label, _) in loop:
                x = x.to("cuda")
                label = label.to("cuda")
                pred = self.deep_model(x)
                self.optim.zero_grad()

                with torch.no_grad():
                    label_hvi = self.hvi_net.trans.HVIT(label)
                    pred_hvi = self.hvi_net.trans.HVIT(pred.clamp(0.0, 1.0))

                hvi_loss = self.L1L(pred_hvi, label_hvi)
                l1_loss = self.L1L(pred, label)
                vgg_loss = self.vggL(pred, label)
                ssim_loss = self.ssimL(pred, label)
                edge_loss = self.edgeL(pred, label)
                final_loss = (
                    l1_loss
                    + 0.5 * hvi_loss
                    + 0.1 * ssim_loss
                    + 0.1 * vgg_loss
                    + 0.4 * edge_loss
                )

                loss_mean += final_loss.item()

                final_loss.backward()

                self.optim.step()
                loop.set_description(f"[{epoch}/{self.args.epoch}]")
                loop.set_postfix(loss=final_loss.item())

            print(
                f"[{epoch}/{self.args.epoch}], avg. loss is {loss_mean / len(train_data_loader)}, learning rate is {self.optim.param_groups[0]['lr']}"
            )
            if epoch % self.args.epoch_val == 0:
                self.deep_model.eval()
                # ssim_, psnr_, uiqm_, uciqe_ = self.validation()
                ssim_, psnr_ = self.validation()

                if psnr_ > best_psnr:
                    torch.save(
                        self.deep_model.state_dict(),
                        os.path.join(self.model_save_path, f"best_model.pth"),
                    )
                    best_psnr = psnr_
                    best_round = {
                        "best epoch": epoch,
                        "best PSNR": best_psnr,
                        "best SSIM": ssim_,
                        # "best UIQM": uiqm_,
                        # "best UCIQE": uciqe_,
                    }
                    with open(
                        os.path.join(self.model_save_path, "records.txt"), "a"
                    ) as f:
                        str_ = "## best round ##\n"
                        for k, v in best_round.items():
                            str_ += f"{k}: {v}. "
                        str_ += "\n####################################"
                        f.write(str_ + "\n")
                with open(os.path.join(self.model_save_path, "records.txt"), "a") as f:
                    # str_ = f"[epoch: {epoch}], PSNR: {psnr_}, SSIM: {ssim_}, UIQM: {uiqm_}, UCIQE: {uciqe_}"
                    str_ = f"[epoch: {epoch}], PSNR: {psnr_}, SSIM: {ssim_}"
                    f.write(str_ + "\n")
                self.deep_model.train()
            self.scheduler.step()

        print("The accuracy of the best round is ", best_round)

    def validation(self):
        self.evaluator.reset()
        val_data_loader = get_loader(
            self.args.val_root,
            self.args.eval_batch_size,
            self.args.datasize,
            train=False,
            resize=self.args.resize,
            num_workers=1,
            shuffle=False,
            pin_memory=True,
        )
        torch.cuda.empty_cache()

        with torch.no_grad():
            loop = tqdm(
                enumerate(val_data_loader), total=len(val_data_loader), leave=False
            )
            for _, (x, label, _) in loop:
                x = x.to("cuda")
                label = (
                    label.numpy().astype(np.float32).transpose(0, 2, 3, 1)
                )  # B, H, W, C
                pred = self.deep_model(x)
                pred = torch.clamp(pred, 0.0, 1.0)
                pred = (
                    pred.data.cpu().numpy().astype(np.float32).transpose(0, 2, 3, 1)
                )  # B, H, W, C

                self.evaluator.evaluation(pred, label)
                loop.set_description("[Validation]")
        # ssim_, psnr_, uiqm_, uciqe_ = self.evaluator.getMean()
        ssim_, psnr_ = self.evaluator.getMean()

        print(
            # "[Validation] SSIM: %.4f, PSNR: %.4f, UIQM: %.4f, UCIQE: %.4f"
            "[Validation] SSIM: %.4f, PSNR: %.4f"
            % (ssim_, psnr_)
            # % (ssim_, psnr_, uiqm_, uciqe_)
        )
        # return ssim_, psnr_, uiqm_, uciqe_
        return ssim_, psnr_


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epoch", type=int, default=100, help="epoch number")
    parser.add_argument("--epoch_val", type=int, default=1, help="training batch size")
    parser.add_argument("--lr", type=float, default=2e-3, help="learning rate")
    parser.add_argument("--train_batch_size", type=int, default=24)
    parser.add_argument("--eval_batch_size", type=int, default=24)
    parser.add_argument(
        "--decay_rate", type=float, default=0.1, help="decay rate of learning rate"
    )  ##
    parser.add_argument(
        "--decay_epoch", type=int, default=50, help="every n epochs decay learning rate"
    )  ##
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--scheduler", type=str, default="cosine")

    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--dataset", type=str, default="UIEB")
    parser.add_argument("--model_name", type=str, default="WWE-UIE")
    parser.add_argument("--save_path", type=str, default="./output/")

    parser.add_argument("--resume", type=str)

    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.training()


if __name__ == "__main__":
    start = time.time()
    seed_everything(9999)

    main()

    end = time.time()
    print("The total training time is:", end - start)
