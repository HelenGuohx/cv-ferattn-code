import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import cv2
from torchlib.attentionnet import AttentionNeuralNet, AttentionGMMNeuralNet
from torchlib.classnet import ClassNeuralNet
from torchlib.datasets.datasets import Dataset
from aug import get_transforms_aug, get_transforms_det
from pytvision.datasets import utility
from pytvision.transforms.aumentation import ObjectImageAndLabelTransform, ObjectImageTransform

fnet = {
    'attnet': AttentionNeuralNet,
    'attgmmnet': AttentionGMMNeuralNet,
    'classnet': ClassNeuralNet,

}


def fer_learner(fname, pathname):
    return FerLearner(fname, pathname)


class FerLearner(object):
    def __init__(self, fname, pathname):
        print('>> Load model ...')
        self.fname = fname
        project = f'../out/{fname}'
        projectname = pathname
        namemodel = "models/model_best.pth.tar"
        pathnamemodel =f"{project}/{projectname}/{namemodel}"
        self.no_cuda = True
        parallel = False
        gpu = 0
        seed = 1

        self.breal = 'real'
        self.classes = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear', 'Contempt']

        self.network = fnet[fname](
            patchproject=project,
            nameproject=projectname,
            no_cuda=self.no_cuda,
            parallel=parallel,
            seed=seed,
            gpu=gpu,
        )

        cudnn.benchmark = True

        # load trained model
        if self.network.load(pathnamemodel) is not True:
            print('>>Error!!! load model')
            assert False

        self.network.net.eval()

    def predict(self, img):
        if not self.no_cuda:
            img = img.cuda()

        if self.fname == 'classnet':
            y = self.network.net(img)
            y = y.detach().numpy()
            return y
        else:
            y, att, g_att, g_ft = self.network.net(img)
            y, att, g_att, g_ft = y.detach().numpy(), att.detach().numpy(), g_att.detach().numpy(), g_ft.detach().numpy()
            return y, att, g_att, g_ft

    @staticmethod
    def transform_image(image, image_size, num_channels):
        image = np.array(image, dtype=np.uint8)
        image = utility.to_channels(image, num_channels)
        obj = ObjectImageTransform(image)
        obj = get_transforms_det(image_size)(obj)
        # tensor is channel x height x width
        # image = image.transpose((2, 0, 1))
        # obj.to_tensor()
        # value = torch.from_numpy(obj.image).float()
        obj.image = obj.image.unsqueeze(0)
        return obj.to_dict()


if __name__ == "__main__":
    # os.environ['DISPLAY'] = ':0'
    learn = fer_learner("ferattn","feratt_attnet_ferattention_attloss_adam_ck_synthetic_filter32_pool_size2_dim32_bbpreactresnet_fold5_000")
    img_cp = cv2.imread("../out/happy.jpg")
    imagesize = 64
    num_channel = 3
    # img_dataset = Dataset([img_cp], num_channels=num_channel, transform=get_transforms_det(imagesize))
    trans_image = learn.transform_image(img_cp, 64, 3)['image']
    yhat, att, g_att, g_ft = learn.predict(trans_image)
    label = np.argmax(yhat, axis=1).squeeze()
    probability = yhat.squeeze()[label]
    prediction = learn.classes[label]
    print(f"prediction: {prediction}, prob:{probability}")



