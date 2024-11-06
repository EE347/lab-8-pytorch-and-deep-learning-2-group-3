import os

import cv2
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class TeamMateDataset(Dataset):

    def __init__(self, n_images=50, train=True):

        if train:
            dataset_type = 'train'
        else:
            dataset_type = 'test'

        subject_0 = os.listdir(f'data/{dataset_type}/0')
        subject_1 = os.listdir(f'data/{dataset_type}/1')

        assert len(subject_0) >= n_images and len(subject_1) >= n_images, f'Number of images in each folder should be {n_images}'

        subject_0 = subject_0[: n_images]
        subject_1 = subject_1[: n_images]

        image_paths = subject_0 + subject_1

        self.dataset = torch.zeros((n_images * 2, 64, 64, 3), dtype=torch.float32)
        self.labels = torch.zeros((n_images * 2), dtype=torch.long)

        for i, image_path in tqdm(enumerate(image_paths), desc="Loading Images", total=n_images * 2, leave=False):

            if i >= n_images:
                subject = 1
            else:
                subject = 0

            image = cv2.imread(f'data/{dataset_type}/{subject}/' + image_path)

            # Resize the image to 64x64
            image = cv2.resize(image, (64, 64))

            # Convert the image to grayscale
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Normalize the image
            image = image / 255.0

            # Add the image to the dataset
            self.dataset[i] = torch.tensor(image)

            # Add the label to the labels tensor
            self.labels[i] = subject


    def __len__(self):
        return self.dataset.shape[0]


    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]