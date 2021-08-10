import os
import numpy as np
from skimage import io
import cv2


# Taken from here https://github.com/zllrunning/face-parsing.PyTorch/issues/12#issuecomment-647879076,
# which is different from the official project https://github.com/switchablenorms/CelebAMask-HQ/tree/master/face_parsing
mask_atts = {0: 'background', 1: 'skin', 2: 'l_brow', 3: 'r_brow', 4: 'l_eye', 5: 'r_eye',
             6: 'eye_g', 7: 'l_ear', 8: 'r_ear', 9: 'ear_r', 10: 'nose', 11: 'mouth',
             12: 'u_lip', 13: 'l_lip', 14: 'neck', 15: 'neck_l', 16: 'cloth', 17: 'hair', 18: 'hat'}


def set_maskpath(x):
    x = x.split('/')
    x[-2] = 'mask_faceparsing'
    x = '/'.join(x)
    x = x.replace('.jpg', '.png')
    return x


def get_listitems(l, ids):
    return list(map(l.__getitem__, ids))


def resize_imx144(x):
    '''
    Function created to isolate the cv2 method, mainly becase skimage.transform.resize changes
    the type of the input array, which can mislead the data flow.
    '''
    return cv2.resize(x, (144, 144))


def load_img(fpath):
    try:
        im = io.imread(fpath)
        return resize_imx144(im)
    except Exception:
        return None


def load_mask(fpath):
    exclude_maskAtts = [0, 7, 8, 9, 14, 15, 16]
    try:
        mask = io.imread(fpath)
        mask = (~ np.isin(mask, exclude_maskAtts)) * 1.
        return resize_imx144(mask)
    except Exception:
        return None


class InputGen:
    def __init__(self, bs=32, impaths=[], shuffle=True, loadsize_factor=50, mode_predict=False, no_mask=False):
        self.impaths = impaths
        self.maskpaths = list(map(set_maskpath, impaths))
        self.nsamples = len(impaths)
        self.bs = bs
        self.mode_predict = mode_predict
        self.no_mask = no_mask

        self.load_size = loadsize_factor * bs
        # self.load_size = 2 * bs
        self.load_idstart = 0
        self.load_idend = self.load_size
        self.load_ids = np.arange(self.load_idstart, self.load_idend)
        self.shuffle = shuffle
        if self.shuffle:
            self.shuffle_samples()

        self.batch_idstart = 0
        self.batch_idend = bs

        self.load_images()

    def load_images(self):
        ims = [load_img(self.impaths[i]) for i in self.load_ids]
        self.X = np.stack(ims) / 255

        if not self.mode_predict or not self.no_mask:
            mask = [load_mask(self.maskpaths[i]) for i in self.load_ids]
            self.Xmask = np.stack(mask).astype(np.float32)

    def shuffle_samples(self):
        np.random.shuffle(self.impaths)
        self.maskpaths = list(map(set_maskpath, self.impaths))

    def reload_cycle(self):
        if self.load_idend >= self.nsamples:
            self.load_idstart = 0
            self.load_idend = self.load_size
            if self.shuffle:
                self.shuffle_samples()
        else:
            self.load_idstart += self.load_size
            self.load_idend = np.min([self.load_idend + self.load_size, self.nsamples])

        self.load_ids = np.arange(self.load_idstart, self.load_idend)
        self.load_images()
        self.batch_idstart = 0
        self.batch_idend = self.bs

    def generator(self, attr=None, factor_attr=1.):
        while True:
            self.X_batch = self.X[self.batch_idstart:self.batch_idend]
            if self.mode_predict:
                if attr is None:
                    yield {'in_image': self.X_batch}
                else:
                    batch_attr = np.stack([attr * factor_attr for _ in range(self.bs)])
                    yield {'in_image': self.X_batch, 'in_attr': batch_attr}
            elif self.no_mask:
                yield {'in_image': self.X_batch}, {'out_image': self.X_batch}
            else:
                self.Xmask_batch = self.Xmask[self.batch_idstart:self.batch_idend]
                yield {'in_image': self.X_batch, 'in_mask': self.Xmask_batch}, \
                      {'out_image': self.X_batch, 'out_mask': self.Xmask_batch}

            self.batch_idstart += self.bs
            self.batch_idend = np.min([self.batch_idend + self.bs, self.load_size])
            if self.batch_idend >= len(self.load_ids):
                self.reload_cycle()

    def get_batch_impaths(self):
        batch_ids = self.load_ids[self.batch_idstart:self.batch_idend].tolist()
        return get_listitems(self.impaths, batch_ids), get_listitems(self.maskpaths, batch_ids)
