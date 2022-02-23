from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os


class AnimalDataset(Dataset):
    def __init__(self, root, file_name, transform=None):
        self.root = root
        self.file_name = file_name
        self.transform = transform
        self.all_file = [os.path.join(self.root,k,i)for k,v in self.file_name.items()  for i in v ]
        self.class_name = [k for k in self.file_name]
        self.id_to_class = {k:v for k,v in enumerate(self.class_name)}
        self.class_to_id = {v:k for k,v in self.id_to_class.items()}
    def __len__(self):
        return len(self.all_file)
    def __getitem__(self, i):
        file = self.all_file[i]
        image = Image.open(file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        class_name = file.split('\\')[-2]
        label = self.class_to_id[class_name]
        # print(file)
        if self.transform is not None:
            image = self.transform(image)
        return image, label
        
        