import os
import sys
from PIL import Image

import numpy as np
import scipy

import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets


class TwilightDuo(Dataset):
    def __init__(self, n_samples, bias, transform=None):
        self.n_samples = n_samples
        self.bias = bias
        self.transform = transform

        self.moon = np.load("/home/ym2380/elastic_net/shortcut/code/data/moon.npy")
        self.star = np.load("/home/ym2380/elastic_net/shortcut/code/data/star.npy")

        moon_topleft = ((46, 60), (39, 70))
        moon_bottomright = ((-53, -39), (-65, -34))
        star_bottomleft = ((-46, -22), (44, 64))
        star_topright = ((55, 78), (-56, -33))

        moon_unbiased = ((-53, 60), (-65, 70))
        star_unbiased = ((-46, 78), (-56, 64))

        n_moon = self.n_samples // 2
        n_star = self.n_samples - n_moon

        moon_X, moon_y = self._gen_data(
            "moon",
            n_moon,
            moon_topleft,
            moon_bottomright,
            moon_unbiased,
        )
        star_X, star_y = self._gen_data(
            "star",
            n_star,
            star_bottomleft,
            star_topright,
            star_unbiased,
        )

        X = np.concatenate((moon_X, star_X), axis=0)
        y = np.concatenate((moon_y, star_y), axis=0)

        permutation_indices = np.random.permutation(self.n_samples)
        self.X = X[permutation_indices]
        self.y = y[permutation_indices]

    def __getitem__(self, idx):
        if self.transform is None:
            data_point = self.X[idx]
        else:
            data_point = self.transform(self.X[idx])
        return data_point, self.y[idx]

    def __len__(self):
        return self.n_samples

    def _gen_data(self, obj_name, n_samples, biased_area1, biased_area2, unbiased_area):
        X = list()
        obj = self.moon if obj_name == "moon" else self.star
        target = 1 if obj_name == "moon" else 0
        y = np.full(n_samples, target)

        for i in range(n_samples):
            if np.random.binomial(1, self.bias):
                if np.random.choice([True, False]):
                    data_point = self._gen_data_point(obj, biased_area1[0], biased_area1[1])
                else:
                    data_point = self._gen_data_point(obj, biased_area2[0], biased_area2[1])
            else:
                data_point = self._gen_data_point(obj, unbiased_area[0], unbiased_area[1])
            X.append(data_point)
        return np.array(X), y

    def _gen_data_point(self, obj, x_interval, y_interval):
        t_x = np.random.randint(x_interval[0], x_interval[1])
        t_y = np.random.randint(y_interval[0], y_interval[1])
        translation_matrix = np.array([[1, 0, t_x],
                                       [0, 1, t_y]], dtype=np.float32)
        img = self._warp_affine(obj, translation_matrix)
        return img

    def _warp_affine(self, img, matrix, output_shape=(200, 200)):
        translation = matrix[:, 2]

        homogeneous_matrix = np.eye(3)
        homogeneous_matrix[:2, :] = matrix

        transformed_img = scipy.ndimage.affine_transform(
            img,
            homogeneous_matrix[:2, :2],
            offset=translation,
            output_shape=output_shape,
            order=1,
            mode="constant",
            cval=0,
        )

        return transformed_img



class ColoredMNIST(datasets.VisionDataset):
    def __init__(self, root, train, mnist_corr, color_corr, transform=None):
        """
        Args:
            root (str): parent directory of data
            train (bool): are data sampled from and used for training or not
            mnist_corr (float): correlation between the mnist digit and the target
            color_corr (float): correlation between the color and the target
            transform: transforms to the image
        """
        super(ColoredMNIST, self).__init__(root=root)
        self.train = train
        self.mnist_corr = mnist_corr
        self.color_corr = color_corr
        self.transform = transform

        self.cmnist_dir = os.path.join(self.root, "cmnist/")
        if not os.path.exists(self.cmnist_dir):
            os.makedirs(self.cmnist_dir)
        self.readme = os.path.join(self.cmnist_dir, "README")
        with open(self.readme, 'a') as file:
            pass

        filename = self._prepare_dataset()
        self.data_list = torch.load(os.path.join(self.cmnist_dir, filename))

    def __getitem__(self, idx):
        img, target = self.data_list[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data_list)

    def _prepare_dataset(self):
        with open(self.readme, 'r') as file:
            count = 0
            for line in file:
                ptfn, mc, cc = line.strip().split()
                tr = ptfn.split('.')[0].rstrip("0123456789")
                if (tr == "train") == self.train:
                    count += 1
                    mc, cc = float(mc), float(cc)
                    if mc == self.mnist_corr and cc == self.color_corr:
                        #print("dataset already exists.")
                        return ptfn

        #print("preparing colored mnist.")
        mnist_dir = os.path.join(self.root, "mnist/")
        if not os.path.join(mnist_dir):
            os.madedirs(mnist_dir)

        if self.train:
            mnist = datasets.MNIST(mnist_dir, train=True, download=True)
        else:
            mnist = datasets.MNIST(mnist_dir, train=False, download=True)

        dataset = list()
        for idx, (img, label) in enumerate(mnist):
            img_array = np.array(img)
            binary_label = 0 if label < 5 else 1

            if np.random.rand() < 1 - self.mnist_corr:
                binary_label = binary_label ^ 1

            color_red = binary_label == 0
            if np.random.rand() < 1 - self.color_corr:
                color_red = not color_red

            colored_arr = self._color_grayscale_arr(img_array, red=color_red)

            dataset.append((colored_arr, binary_label))

        pt_filename = "{}{}.pt".format("train" if self.train else "test", count + 1)
        torch.save(dataset, os.path.join(self.cmnist_dir, pt_filename))
        #print(f"{pt_filename} is ready.")
        args_line = f"{pt_filename} {self.mnist_corr} {self.color_corr}"
        with open(self.readme, 'a') as file:
            file.write(args_line + '\n')
        return pt_filename
    
    def _color_grayscale_arr(self, arr, red):
        assert arr.ndim == 2
        dtype = arr.dtype
        h, w = arr.shape
        arr = np.reshape(arr, [h, w, 1])
        if red:
            arr = np.concatenate([
                arr, 
                np.zeros((h, w, 2), dtype=dtype)], axis=2)
        else:
            arr = np.concatenate([
                np.zeros((h, w, 1), dtype=dtype),
                arr,
                np.zeros((h, w, 1), dtype=dtype)], axis=2)
        return arr
