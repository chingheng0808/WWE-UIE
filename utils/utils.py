import re
import matplotlib.pyplot as plt
import os


def plot_curve(file_path, save=False, model_info=None):
    # read file
    with open(file_path, "r") as f:
        lines = f.readlines()

    epoch_list = []
    psnr_list = []

    best_epochs = []
    best_psnrs = []

    for line in lines:
        line = line.strip()
        m = re.search(
            r"\[epoch:\s*(\d+)\],\s*PSNR:\s*([\d.]+),\s*SSIM:\s*([\d.]+)", line
        )
        if m:
            epoch = int(m.group(1))
            psnr = float(m.group(2))
            epoch_list.append(epoch)
            psnr_list.append(psnr)

        m2 = re.search(
            r"best epoch:\s*(\d+)\. best PSNR:\s*([\d.]+)\. best SSIM:\s*([\d.]+)", line
        )
        if m2:
            best_epoch = int(m2.group(1))
            best_psnr = float(m2.group(2))
            best_epochs.append(best_epoch)
            best_psnrs.append(best_psnr)

    # remove duplicates
    unique_best = {}
    for be, bi in zip(best_epochs, best_psnrs):
        unique_best[be] = bi

    best_x = list(unique_best.keys())
    best_y = list(unique_best.values())

    # plotting
    plt.figure(figsize=(10, 6))
    plt.plot(
        epoch_list,
        psnr_list,
        label="PSNR",
        marker="o",
        markersize=5,
        linestyle="-",
        color="blue",
    )
    plt.scatter(best_x, best_y, s=150, color="red", marker="*", label="Best Round")

    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    model_name = file_path.split("/")[-4]
    dataset = file_path.split("/")[-3]
    plt.title(f"[{model_name}/{dataset}] Validation PSNR over Epochs")
    plt.legend()
    plt.grid(True)

    if model_info is not None:
        # model_info need to be a dictionary
        info_text = f"Params: {model_info['params']}, FLOPs: {model_info['flops']}\n PSNR: {model_info['psnr']}, SSIM: {model_info['ssim']} \n Time: {model_info['time']}"

        plt.text(
            0.98,
            0.02,
            info_text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(facecolor="white", alpha=0.5),
        )
    if save:
        save_path = f'{"/".join(file_path.split("/")[:-1])}/plot.png'
        plt.savefig(save_path)
    plt.show()


def store_restored(batch_pred, batch_label, fns, store_fd):
    # input: (B, H, W, C): numpy.array()
    store_fd = os.path.join(store_fd, "test_preds")
    if not os.path.exists(store_fd):
        os.makedirs(store_fd)
    for i in range(batch_pred.shape[0]):
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(batch_pred[i])
        plt.axis("off")
        plt.title("Prediction")
        plt.subplot(1, 2, 2)
        plt.imshow(batch_label[i])
        plt.axis("off")
        plt.title("Ground truth")

        plt.tight_layout()

        plt.savefig(os.path.join(store_fd, os.path.basename(fns[i])))
        plt.close()
    return
