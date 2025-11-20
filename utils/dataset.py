import torch.utils.data as data
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
from torchvision.transforms import ToTensor


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".bmp"])


def load_img(filepath):
    img = Image.open(filepath).convert("RGB")
    return img


def rescale_img(img_in, size: tuple):
    img_in = img_in.resize(size, Image.BILINER)
    return img_in


def get_patch(img_in, img_tar, patch_size, scale=1, ix=-1, iy=-1):
    (ih, iw) = img_in.size

    patch_mult = scale
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))
    img_tar = img_tar.crop((ty, tx, ty + tp, tx + tp))

    info_patch = {"ix": ix, "iy": iy, "ip": ip, "tx": tx, "ty": ty, "tp": tp}

    return img_in, img_tar, info_patch


def augment(img_in, img_tar, flip_h=True, rot=True):
    info_aug = {"flip_h": False, "flip_v": False, "trans": False}

    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        info_aug["flip_h"] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            info_aug["flip_v"] = True
        if random.random() < 0.5:
            rot_ang = random.choice([90, 180, 270])
            img_in = img_in.rotate(rot_ang)
            img_tar = img_tar.rotate(rot_ang)
            info_aug["trans"] = True

    return img_in, img_tar, info_aug


class DatasetFromFolder(data.Dataset):
    def __init__(
        self, data_root, data_size, transform=ToTensor(), train=True, resize=False
    ):
        super(DatasetFromFolder, self).__init__()
        assert data_size is not None, "parameter [full_size] cannot be None"
        self.train = train
        data_filenames = [
            join(data_root, "input", x)
            for x in listdir(join(data_root, "input"))
            if is_image_file(x)
        ]
        data_filenames.sort()
        self.data_filenames = data_filenames
        label_filenames = [
            join(data_root, "GT", x)
            for x in listdir(join(data_root, "GT"))
            if is_image_file(x)
        ]
        label_filenames.sort()
        self.label_filenames = label_filenames
        self.data_size = data_size
        self.transform = transform
        self.resize = resize

    def __getitem__(self, index):
        target = load_img(self.label_filenames[index])
        input = load_img(self.data_filenames[index])
        _, file = os.path.split(self.label_filenames[index])

        if target.size != input.size:
            target = target.resize((input.size[0], input.size[1]), Image.BILINEAR)
        if self.resize:
            input, target = input.resize(
                (self.data_size, self.data_size), Image.BILINEAR
            ), target.resize((self.data_size, self.data_size), Image.BILINEAR)

        if self.train:
            input, target, _ = get_patch(input, target, self.data_size)
            input, target, _ = augment(input, target)

        if self.transform:
            input = self.transform(input)
            target = self.transform(target)

        return input, target, file

    def __len__(self):
        return len(self.label_filenames)


class DatasetFromFolder_NR(data.Dataset):
    def __init__(
        self, data_root, transform=ToTensor(), data_size=256, resize=True
    ):
        super(DatasetFromFolder_NR, self).__init__()

        data_filenames = [
            join(data_root, x) for x in os.listdir(data_root) if is_image_file(x)
        ]

        data_filenames.sort()
        self.data_filenames = data_filenames
        self.transform = transform
        self.data_size = data_size
        self.resize = resize

    def __getitem__(self, index):
        input_img = load_img(self.data_filenames[index])
        _, file = os.path.split(self.data_filenames[index])

        if self.resize:
            w, h = input_img.size
            if w < h:
                new_w = self.data_size
                new_h = int(h * self.data_size / w)
            else:
                new_h = self.data_size
                new_w = int(w * self.data_size / h)
            input_img = input_img.resize((new_w, new_h), Image.BILINEAR)

        # comfirm image size is 4's multiple
        w, h = input_img.size
        new_w = (w // 4) * 4
        new_h = (h // 4) * 4
        if (new_w != w) or (new_h != h):
            input_img = input_img.crop((0, 0, new_w, new_h))  # from left-top

        if self.transform:
            input_img = self.transform(input_img)

        return input_img, file

    def __len__(self):
        return len(self.data_filenames)


def get_loader(
    data_root,
    batchsize,
    data_size,
    train=True,
    resize=False,
    num_workers=1,
    shuffle=True,
    pin_memory=True,
    non_ref=False,
):
    if non_ref:
        dataset = DatasetFromFolder_NR(
            data_root=data_root, data_size=data_size, resize=resize
        )
    else:
        dataset = DatasetFromFolder(
            data_root=data_root, data_size=data_size, train=train, resize=resize
        )
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return data_loader
