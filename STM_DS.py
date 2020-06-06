import os
import os.path as osp
import numpy as np
from PIL import Image
import glob
import torch
import torchvision
from torch.utils import data
import torchvision.transforms as TF

# from dataset import transforms as mytrans
# import myutils

#
# class STM_USER_Test_DS(data.Dataset):
#
#     def __init__(self, root, dataset_file='videos.txt', max_obj_n=5, output_size=None):
#         self.root = root
#
#         dataset_path = os.path.join(root, dataset_file)
#         self.dataset_list = list()
#         with open(os.path.join(dataset_path), 'r') as lines:
#             for line in lines:
#                 dataset_name = line.strip()
#                 self.dataset_list.append(dataset_name)
#
#         if output_size:
#             self.resize = TF.Resize(output_size)
#         self.to_tensor = TF.ToTensor()
#         self.to_onehot_tensor = mytrans.ToOnehotTensor(max_obj_n)
#
#     def __len__(self):
#         return len(self.dataset_list)
#
#     def __getitem__(self, idx):
#
#         dataset_name = self.dataset_list[idx]
#         img_dir = os.path.join(self.root, 'JPEGImages', dataset_name)
#         mask_dir = os.path.join(self.root, 'Annotations', dataset_name)
#
#         img_list = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
#         mask_list = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
#
#         first_mask = myutils.load_image_in_PIL(mask_list[0], 'P')
#         if self.resize:
#             first_mask = np.asarray(self.resize(first_mask))
#         else:
#             first_mask = np.asarray(first_mask)
#         h, w = first_mask.shape
#         obj_n = first_mask.max() + 1
#         video_len = len(img_list)
#
#         frames = torch.zeros((video_len, 3, h, w), dtype=torch.float)
#         masks = torch.zeros((video_len, obj_n, h, w), dtype=torch.float)
#
#         for i in range(video_len):
#             img = myutils.load_image_in_PIL(img_list[i], 'RGB')
#             img = self.resize(img)
#             frames[i] = self.to_tensor(img)
#
#             try:
#                 mask = myutils.load_image_in_PIL(mask_list[i], 'P')
#                 mask = self.resize(mask)
#                 mask, _ = self.to_onehot_tensor(mask)
#                 masks[i] = mask[:obj_n]
#             except IndexError as e:
#                 pass
#
#         frames = frames.transpose(0, 1)
#         masks = masks.transpose(0, 1)
#         info = {
#             'name': dataset_name,
#             'num_frames': video_len,
#         }
#
#         return frames, masks, obj_n - 1, info


class STM_DAVIS_DS(data.Dataset):
    # for multi object, do shuffling

    def __init__(self, root, imset='2017/train.txt', resolution='480p', single_object=False):
        self.root = root
        self.mask_dir = os.path.join(root, 'Annotations', resolution)
        self.mask480_dir = os.path.join(root, 'Annotations', '480p')
        self.image_dir = os.path.join(root, 'JPEGImages', resolution)
        _imset_dir = os.path.join(root, 'ImageSets')
        _imset_f = os.path.join(_imset_dir, imset)

        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        self.size_480p = {}
        with open(os.path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                if _video == 'bike-packing':
                    pass
                self.videos.append(_video)
                self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))
                _mask = np.array(Image.open(os.path.join(self.mask_dir, _video, '00000.png')).convert("P"))
                self.num_objects[_video] = np.max(_mask)
                self.shape[_video] = np.shape(_mask)
                _mask480 = np.array(Image.open(os.path.join(self.mask480_dir, _video, '00000.png')).convert("P"))
                self.size_480p[_video] = np.shape(_mask480)

        self.K = 11
        self.single_object = single_object

    def __len__(self):
        return len(self.videos)

    def To_onehot(self, mask):
        M = np.zeros((self.K, mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for k in range(self.K):
            M[k] = (mask == k).astype(np.uint8)
        return M

    def All_to_onehot(self, masks):
        Ms = np.zeros((self.K, masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
        for n in range(masks.shape[0]):
            Ms[:, n] = self.To_onehot(masks[n])
        return Ms

    def __getitem__(self, index):
        video = self.videos[index]
        # video = 'scooter-black'
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames[video]
        info['size_480p'] = self.size_480p[video]

        N_frames = np.empty((self.num_frames[video],) + self.shape[video] + (3,), dtype=np.float32)
        # N_masks = np.empty((self.num_frames[video],) + self.shape[video], dtype=np.uint8)
        N_masks = np.empty((1,) + self.shape[video], dtype=np.uint8)
        for f in range(self.num_frames[video]):
            img_file = os.path.join(self.image_dir, video, '{:05d}.jpg'.format(f))
            N_frames[f] = np.array(Image.open(img_file).convert('RGB')) / 255.
            try:
                mask_file = os.path.join(self.mask_dir, video, '{:05d}.png'.format(f))
                N_masks[f] = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
            except:
                # print('a')
                # N_masks[f] = 255
                pass

        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
        if self.single_object:
            N_masks = (N_masks > 0.5).astype(np.uint8) * (N_masks < 255).astype(np.uint8)
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(1)])
            return Fs, Ms, num_objects, info
        else:
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(self.num_objects[video])])
            return Fs, Ms, num_objects, info


if __name__ == '__main__':
    pass