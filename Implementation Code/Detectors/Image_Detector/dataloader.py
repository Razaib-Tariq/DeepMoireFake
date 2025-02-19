import os
from torch.utils.data import Dataset
from PIL import Image
import glob
import numpy as np



class FaceDataset(Dataset):
    def __init__(self, datafolder, transform, use_bgr=False, sampling_rate=1):
        self.use_bgr = use_bgr
        print("Gather all images")
        if isinstance(datafolder, list):
            self.photo_path = []
            self.labels = []
            for p in datafolder:
                print(p.split('/')[-2].upper())
                all_frames = glob.glob(p + '/*/*.png') # First * is video name, second * is frame name
                self.photo_path += all_frames
                if "real" in p.lower():
                    self.labels += [0]*len(all_frames)
                else:
                    self.labels += [1]*len(all_frames)
        else:
            print(datafolder.split('/')[-2].upper())
            self.photo_path = glob.glob(datafolder + '/*/*.png')
            self.labels = [1] * len(self.photo_path)

        self.photo_path = self.photo_path[::sampling_rate]
        self.labels = self.labels[::sampling_rate]
        print("Test lb:", sum(self.labels), self.photo_path[-1], self.labels[-1])

        print("Total images: ",  len(self.photo_path))
        self.video_id = [self.photo_belong_to_video_ID(u) for u in  self.photo_path ]
        self.ID2Num = dict(zip(list(set(self.video_id)), range(len(list(set(self.video_id))))))

        print("Number of videos: ", len(list(set(self.video_id))))

        for p in self.photo_path:
            assert os.path.exists(p), f"Path not exists: {p}"
        self.transforms = transform
 
    def __len__(self):
        return len(self.photo_path)
    
    def photo_belong_to_video_ID(self, video_path):
        # Get the absolute path of the directory
        # directory_path = os.path.abspath(os.path.dirname(video_path))
        # # Get the name of the parent directory
        # parent_dir_name = os.path.basename(directory_path)
        
        # return parent_dir_name
        return '_'.join(video_path.split('/')[:-1])
    
    def __getitem__(self, item):

        img_path = self.photo_path[item]
        videoID = self.ID2Num[self.video_id[item]]
        lb = self.labels[item]
        img = Image.open(img_path)
        if self.use_bgr:
            img = Image.fromarray(np.array(img)[:,:,::-1])
            img = self.transforms(img)  * 255.0
            return img, lb, videoID
        img = self.transforms(img)
        # print(img.size())
        return img, lb, videoID