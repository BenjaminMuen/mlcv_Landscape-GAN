import os

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from PIL import Image

class Landscape_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # get all images from the root directory
        self.image_files = [os.path.join(root_dir, file) for file in os.listdir(root_dir) if file.endswith(('.png', '.jpg', 'jpeg'))]

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        image_path = self.image_files[index]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        return image

# default data augumentation consists of resizing the images to a Resolution of 64x64 pixels, converting them to tensors and normalizing them
default_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def get_dataloader(root_dir, batch_size, n_workers, transform=None):
    if transform is None:
        transform = default_transform

    dataset = Landscape_Dataset(root_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, persistent_workers=True)

    return dataloader