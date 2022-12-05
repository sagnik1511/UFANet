import os
import pandas as pd
from PIL import Image
from glob import glob
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class UBIrisDataset(Dataset):
    def __init__(self,
                 meta_df_path,
                 real_width=300,
                 mask_width=116,
                 transform=None,
                 storage="local"):
        super().__init__()

        self.meta_df = pd.read_csv(meta_df_path)
        self.real_width = real_width
        self.mask_width = mask_width
        self.transform = transform
        self.storage = storage
        self.process_metadata()

    def __len__(self):
        return len(self.meta_df)

    def process_metadata(self):
        if self.storage == "local":
            self.meta_df["real"] = self.meta_df["real"].apply(self.filter_path)
            self.meta_df["mask"] = self.meta_df["mask"].apply(self.filter_path)

    @staticmethod
    def filter_path(path):
        paths = os.path.normpath(path).split(os.path.sep)[2:]
        path = os.path.join(*paths)
        return path

    def __getitem__(self, index):
        image_path = self.meta_df.loc[index, "real"]
        image = self.load_image(image_path)
        mask_path = self.meta_df.loc[index, "mask"]
        mask = self.load_image(mask_path, real=False)
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        else:
            image = ToTensor()(image)
            mask = ToTensor()(mask)
        return image, mask

    def load_image(self, path, real=True):
        image = Image.open(path)
        if real:
            image = image.resize((self.real_width, self.real_width))
        else:
            image = image.resize((self.mask_width, self.mask_width))
        return image


class UBIrisDatasetTest(Dataset):
    def __init__(self,
                 root_dir,
                 width=300):
        super().__init__()

        self.root_dir = root_dir
        self.width = width
        self.files = glob(f"{root_dir}/*.tiff")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image_path = self.files[index]
        image = self.load_image(image_path)
        image = ToTensor()(image)
        return image

    def load_image(self, path):
        image = Image.open(path)
        image = image.resize((self.width, self.width))
        return image
