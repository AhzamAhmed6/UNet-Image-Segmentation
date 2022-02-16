from cProfile import label
import glob
import os
import imageio
from PIL import Image
import torch.utils.data as data
from DataPipeline.TrainTtransform import get_train_transform




class DataLoaderSegmentation(data.Dataset):
  def __init__(self, folder_path, transform=None):
    super(DataLoaderSegmentation, self).__init__()
    self.img_files = glob.glob(os.path.join(folder_path,'images','*.jpg'))
    # print(self.img_files)
    self.mask_files = []
    for img_path in self.img_files:
      self.mask_files.append(os.path.join(folder_path,'masks',os.path.basename(img_path.split('.')[0]+'_mask.jpg')))
    self.transforms = get_train_transform() 

  
  def __getitem__(self, index):
    img_path = self.img_files[index]
    mask_path = self.mask_files[index]
    data = imageio.imread(img_path)
    label = imageio.imread(mask_path)

    # img = self.transforms(data)
    # mask = self.transforms(label)
    
    augmented = self.transforms(image=data, mask=label)
    img = augmented['image']
    mask = augmented['mask']
    mask = mask[0].permute(2, 0, 1)
    return (img,mask)
    # return torch.from_numpy(data).float(), torch.from_numpy(label).float()

  
  def __len__(self):
    if len(self.img_files) == len(self.mask_files):
      return len(self.img_files)

  def name(self, index):
    return self.img_files[index], self.mask_files[index]