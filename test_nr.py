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
from utils.uranker.uranker_utils import build_model, get_option


class Tester(object):
    def __init__(self, args):
        self.args = args

        self.deep_model = myModel(
            in_channels=3, feature_channels=32, use_white_balance=True
        )

        options = get_option(r"utils/uranker/URanker.yaml")
        options["model"]["resume_ckpt_path"] = r"utils/uranker/URanker_ckpt.pth"
        self.uranker_model = build_model(options["model"])
        self.uranker_model = self.uranker_model.cpu()
        self.uranker_model.eval()

        self.evaluator = Evaluator(no_ref=True, uranker_model=self.uranker_model)

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

        args.uccs = False
        if args.dataset == "CH60":
            args.test_root = "/ssd6/UnderWaterDataset/test_unpaired_img/challenging-60/"
        elif args.dataset == "EUVP":
            args.test_root = (
                "/ssd6/UnderWaterDataset/test_unpaired_img/EUVP/validation/"
            )
        elif args.dataset == "U45":
            args.test_root = "/ssd6/UnderWaterDataset/test_unpaired_img/U45/"
        elif args.dataset == "UCCS":
            args.test_root = "/ssd6/UnderWaterDataset/test_unpaired_img/UCCS/"
            args.uccs = True

        self.dataloader = get_loader(
            self.args.test_root,
            1,
            256,
            train=False,
            resize=True,
            num_workers=1,
            shuffle=False,
            pin_memory=True,
            non_ref=True,
            uccs=self.args.uccs,
        )
        # print(len(self.dataloader))

    def testing(self):
        self.evaluator.reset()
        torch.cuda.empty_cache()

        with torch.no_grad():
            loop = tqdm(
                enumerate(self.dataloader), total=len(self.dataloader), leave=False
            )
            for _, (x, fn) in loop:
                x = x.to("cuda")

                # _,pred,_ = self.deep_model(x)
                pred = self.deep_model(x)
                pred = torch.clamp(pred, 0.0, 1.0)
                pred = (
                    pred.data.cpu().numpy().astype(np.float32).transpose(0, 2, 3, 1)
                )  # B, H, W, C

                self.evaluator.evaluation(pred, None)
                if self.args.store:
                    if not os.path.exists(
                        "/".join(self.args.ckpt.split("/")[:-1])
                        + f"/pred/{self.args.dataset}"
                    ):
                        os.makedirs(
                            "/".join(self.args.ckpt.split("/")[:-1])
                            + f"/pred/{self.args.dataset}"
                        )
                    Image.fromarray((pred[0] * 255).astype(np.uint8)).save(
                        "/".join(self.args.ckpt.split("/")[:-1])
                        + f"/pred/{self.args.dataset}"
                        + "/"
                        + fn[0]
                        + ".png"
                    )
                loop.set_description("[Testing]")
        uiqm_, uciqe_, uranker_ = self.evaluator.getMean()

        print(
            "[Testing] UIQM: %.4f, UCIQE: %.4f, URanker: %.4f"
            % (uiqm_, uciqe_, uranker_)
        )
        # return ssim_, psnr_, uiqm_, uciqe_
        dummy = torch.randn(1, 3, 256, 256).cuda()
        flops, params = profile(self.deep_model, inputs=(dummy,))
        flops, params = clever_format([flops, params], "%.3f")
        print(f"Params: {params}, FLOPs: {flops}")

        return


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        type=str,
    )
    parser.add_argument("--dataset", type=str, default="CH60")
    parser.add_argument("--store", action="store_true", default=False)

    args = parser.parse_args()

    tester = Tester(args)

    start = time.time()

    tester.testing()

    end = time.time()
    print("Testing time:", end - start, "sec")


if __name__ == "__main__":
    main()
