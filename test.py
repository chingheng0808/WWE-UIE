import argparse
import os
import time
import numpy as np

import torch
from tqdm import tqdm
from thop import profile, clever_format
from PIL import Image

from utils.dataset import get_loader
from model import myModel
from utils.metrics import Evaluator
from utils.utils import store_restored


class Tester(object):
    def __init__(self, args):
        self.args = args

        self.evaluator = Evaluator()

        self.deep_model = myModel(
            in_channels=3, feature_channels=32, use_white_balance=True
        )

        if os.path.isfile(args.ckpt):
            checkpoint = torch.load(args.ckpt)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)
        else:
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.ckpt))

        self.deep_model = self.deep_model.to("cuda")
        self.deep_model.eval()

        if args.dataset == "EUVP-d":  # 256x256
            args.test_root = "/ssddd/chingheng/UnderWaterDataset/EUVP-Dark//test"
            args.datasize = 256
            args.resize = False
        elif args.dataset == "EUVP-s":  # 320x240
            args.test_root = "/ssddd/chingheng/UnderWaterDataset/EUVP-Scene/test/"
            args.datasize = 256
            args.resize = False
        elif args.dataset == "UIEB":  # 1280x~800 => 256x256
            args.test_root = "/ssddd/chingheng/UnderWaterDataset/UIEB/test/"
            args.datasize = 256
            args.resize = True
        elif args.dataset == "UFO":  # 320x240
            args.test_root = "/ssddd/chingheng/UnderWaterDataset/UFO-120/test/"
            args.datasize = 256
            args.resize = False
        elif args.dataset == "LSUI":  # 720x405 => 256x256
            args.test_root = "/ssddd/chingheng/UnderWaterDataset/LSUI/test/"
            args.datasize = 256
            args.resize = True

        self.dataloader = get_loader(
            self.args.test_root,
            self.args.test_batch_size,
            self.args.datasize,
            train=False,
            resize=args.resize,
            num_workers=1,
            shuffle=False,
            pin_memory=True,
        )

    def testing(self):
        self.evaluator.reset()
        torch.cuda.empty_cache()

        with torch.no_grad():
            loop = tqdm(
                enumerate(self.dataloader), total=len(self.dataloader), leave=False
            )
            for _, (x, label, fn) in loop:
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
                if not os.path.exists(
                    "/".join(self.args.ckpt.split("/")[:-1]) + "/pred"
                ):
                    os.makedirs("/".join(self.args.ckpt.split("/")[:-1]) + "/pred")
                store_restored(
                    pred, label, fn, "/".join(self.args.ckpt.split("/")[:-1])
                )
                loop.set_description("[Testing]")
        # ssim_, psnr_, uiqm_, uciqe_ = self.evaluator.getMean()
        ssim_, psnr_ = self.evaluator.getMean()

        print(
            # "[Testing] SSIM: %.4f, PSNR: %.4f, UIQM: %.4f, UCIQE: %.4f"
            "[Testing] SSIM: %.4f, PSNR: %.4f"
            % (ssim_, psnr_)
            # % (ssim_, psnr_, uiqm_, uciqe_)
        )
        with open("/".join(self.args.ckpt.split("/")[:-1]) + "/result.txt", "w") as f:
            f.write("[Testing] SSIM: %.4f, PSNR: %.4f" % (ssim_, psnr_))
        # return ssim_, psnr_, uiqm_, uciqe_
        dummy = torch.randn(1, 3, self.args.datasize, self.args.datasize).cuda()
        flops, params = profile(self.deep_model, inputs=(dummy,))
        flops, params = clever_format([flops, params], "%.3f")
        model_info = {
            "params": params,
            "flops": flops,
            "ssim": "%.4f" % ssim_,
            "psnr": "%.4f" % psnr_,
        }
        print(f"Params: {params}, FLOPs: {flops}")

        return model_info


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        type=str,
    )
    parser.add_argument("--dataset", type=str, default="UIEB")
    parser.add_argument("--test_batch_size", type=int, default=4)

    args = parser.parse_args()

    tester = Tester(args)

    start = time.time()

    model_info = tester.testing()

    end = time.time()
    print("Testing time:", end - start, "sec")
    model_info["time"] = end - start


if __name__ == "__main__":
    main()
