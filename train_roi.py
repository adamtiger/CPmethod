from dataloader_utils import FolderStructureBuilder, ModelTypes, Visualizer
from preproc import Proc, AugmentwithPaddedCenterCrop
from torch.utils.data import Dataset, DataLoader
from training_utils import One2OneOptimizer
from datatransform_utils import RoiBox
from dataloader_utils import HeartPart
import torchvision.models as models
from utils import progress_bar
from torch import optim
import torch.nn as nn
import numpy as np
import config
import pickle
import torch
import os


class ResNet18roi(nn.Module):

    def __init__(self):
        super(ResNet18roi, self).__init__()

        # creating the network architecture
        self.conv = nn.Conv2d(1, 3, (1, 1), 1)
        self.resnet = models.resnet18(num_classes=128)

        self.linear = nn.Linear(128, 2)
        self.prelu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        temp = self.prelu(self.conv(x))
        temp = self.prelu(self.resnet(temp))
        temp = self.sigmoid(self.linear(temp))
        return temp.view(-1, 2)
    def save(self, path):
        torch.save(self, path)

    @classmethod
    def load(cls, state_dict, device):
        model = cls()
        model.load_state_dict(state_dict)
        return model.to(device)


class ROITrainLoader:
    """
    This train loader is for loading the following data pairs:
    - image: preprocessed and augmented
    - roi: center and box (height, width)
    """

    def __init__(self, folder, device, batch_size=32, validation_ratio=0.2):
        """
        This class creates train loaders for binary mask or
        dense segmentation based learning algorithms.
        :param folder: Contains subfolders with images: raw mri, real left, real right segm. in binary form
        :param device: a torch.device for the gpu or cpu
        :param validation_ratio: The ratio of the validation samples.
        """
        self.folder = folder
        self.device = device
        self.batch_size = batch_size
        self.validation_ratio = validation_ratio

        self.hash_original_predicted = {}  # if the prediction happens to calculate the volumes this is required

        self._to_ram()
        self._split_data()
        self.train_imgs, self.train_rois = self._unfold_samples(self.train_samples)
        self.val_imgs, self.val_rois = self._unfold_samples(self.val_samples)
    
    def _to_ram(self):
        """
        Reads the serialized binary samples into the RAM.
        """
        self.samples = []

        samples_as_pickle = os.listdir(self.folder)
        for cntr, sample_as_pickle in enumerate(samples_as_pickle, 1):
            path = os.path.join(self.folder, sample_as_pickle)
            with open(path, 'br') as f:
                control_sample = pickle.load(f)

            self.samples.append(control_sample)
            progress_bar(cntr, len(samples_as_pickle), 10)
    
    def _split_data(self):
        # new datasets
        self.train_samples = []
        self.val_samples = []
        # shuffle the samples
        np.random.shuffle(self.samples)
        # split the data
        n = len(self.samples)
        split_idx = round(n * self.validation_ratio)
        self.train_samples = self.samples[split_idx:]
        self.val_samples = self.samples[:split_idx]

    def _unfold_samples(self, samples):
        images = []
        rois = []
        for sample in samples:
            for slice in sample.image_contour_3s:
                part = HeartPart.LN if config.side == 'left' else HeartPart.RN
                if part in sample.image_contour_3s[slice]:
                    for phase in sample.image_contour_3s[slice][part]:
                        image = sample.image_contour_3s[slice][part][phase]['image']
                        roi = sample.image_contour_3s[slice][part][phase]['mask']  # roi
                        images.append(image)
                        rois.append(roi)
        return images, rois

    # Inner class for wrapping the data in a dataset
    class ROIDataset(Dataset):
        """
        Dataset for wrapping the raw images and the corresponding binaries.
        This makes possible to use DataLoader which has several useful properties.
        """

        def __init__(self, device, images, rois):
            """
            device - torch.device, gpu or cpu
            indices - list of image_ids of the original images
            images - list of all the images
            rois - list of roi matrices
            """
            self.device = device
            self.images = images
            self.rois = rois
            self.augment = AugmentwithPaddedCenterCrop((224, 224), (-10, 11), (-10, 11), 0.001).preprocess
        
        def roi_to_contour(self, roi):
            """
            roi - numpy matrix with shape (4), cx, cy, height, width
            """
            contour = np.zeros((1, 2), dtype=np.float32)
            contour[0] = roi[0:2]
            return contour
        
        def contour_to_roi(self, contour):
            """
            contour - numpy matrix with shape (1, 2)
            """
            roi = np.zeros((4), dtype=np.float32)
            roi[0] = contour[0, 0]
            roi[1] = contour[0, 1]
            roi[2] = 0.0
            roi[3] = 0.0
            return roi

        def __len__(self):
            return len(self.images)

        def __getitem__(self, index):
            idx = index
            roi_as_contour = self.roi_to_contour(self.rois[idx])
            img, contours = self.augment(self.images[idx], [roi_as_contour])
            roi = self.contour_to_roi(contours[0])[0:2]  # now only the center is important
            sample = {
                'image_id': idx,
                'image': torch.tensor(img, dtype=torch.float, device=self.device).unsqueeze(0),
                'target': torch.tensor(roi, dtype=torch.float, device=self.device)
            }
            return sample

    def get_trainloader(self):
        """
        Creates a data loader to iterate through the right training part of the dataset.
        """
        dataset = self.ROIDataset(self.device, self.train_imgs, self.train_rois)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def get_validationloader(self):
        """
        Creates a data loader to iterate through the right test part of the dataset.
        """
        dataset = self.ROIDataset(self.device, self.val_imgs, self.val_rois)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


class Parameters:

    def __init__(self):
        self.data_folder = ""
        self.batch_size = None
        self.val_ratio = None
        self.epochs = None
        self.lr = None
        self.device = None
        self.criterion = None
        self.optimizer = None
    
    def serialize(self, path):
        with open(path, 'wb') as pck:
            pickle.dump(self, pck)

class Results:

    def __init__(self, file_name, visual_folder, weight_folder):
        self.c_val_pred = []
        self.c_train_loss = []
        self.c_val_loss = None
        self.weight_dict = {}
        self.images = []
        self.file_name = file_name
        self.weight_folder = weight_folder
        self.vszr = Visualizer(visual_folder)

    def serialize(self):
        with open(self.file_name, 'wb') as pck:
            pickle.dump(self, pck)
    
    def create_checkpoint(self, index, model, optimizer):
        name = "checkpoint" + str(index) + ".pt"
        path = os.path.join(self.weight_folder, name)
        checkpoint = {
            "index": index,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }
        torch.save(checkpoint, path)

    def cache_val_prediction(self, target, predicted):
        for idx in range(target.shape[0]):
            trg = target[idx] * 224
            prd = predicted[idx] * 224
            distance = np.linalg.norm(trg - prd)
            self.c_val_pred.append(distance)

    def cache_train_loss(self, loss):
        self.c_train_loss.append(loss)
        if len(self.c_train_loss) > 10:
            del self.c_train_loss[0]

    def cache_val_loss(self, loss_mean):
        self.c_val_loss = loss_mean

    def cache_model_weights(self, model):
        self.weight_dict = model.state_dict()

    def visualize_hystogram(self, name, data):
        self.vszr.visualize_hystogram(name, data)

    def visualize_indices(self, name, data):
        self.vszr.visualize_indices(name, data)

    def visualize_graph(self, model, input_data):
        self.vszr.visualize_network(model, input_data)

    def cache_image(self, image, target, predicted):
        self.images.append((image / (image.max() + image.min()), target, predicted))

    def send2visualizer(self, global_counter):
        if len(self.c_val_pred) > 0:
            validation_dice = np.array(self.c_val_pred).mean()
            self.vszr.visualize_dices([0.0, validation_dice], global_counter)
            self.c_val_pred.clear()

        if len(self.c_train_loss) > 0 and not(self.c_val_loss is None):
            training_loss = np.array(self.c_train_loss).mean()
            validation_loss = self.c_val_loss
            self.vszr.visualize_loss([training_loss, validation_loss], global_counter)
            self.c_train_loss.clear()
            self.c_val_loss = None

        if len(self.weight_dict) > 0:
            for name, weight in self.weight_dict.items():
                self.vszr.visualize_weights(name, weight, global_counter)
            self.weight_dict.clear()

        if len(self.images) > 0:
            h, w = self.images[0][0].shape
            summary_image = np.zeros((3, h, len(self.images) * w), dtype=np.uint8)
            for col, tercier in enumerate(self.images):
                tercier = list(tercier)
                imageRGB = np.repeat(np.expand_dims(tercier[0], axis=0), repeats=3, axis=0)
                imageRGB[0, :, :] = 255 * RoiBox.roicenter2image(imageRGB[0, :,:], tercier[1] * 224)
                imageRGB[1, :, :] = 255 * RoiBox.roicenter2image(imageRGB[1, :,:], tercier[2] * 224)
                summary_image[:, :, col * w: (col + 1) * w] = imageRGB.astype(np.uint8)
            self.vszr.visualize_image(summary_image, global_counter)
            self.images.clear()


def create_loaders(loader):
    return loader.get_trainloader(), loader.get_validationloader()


# --------------------------------------
# ROI estimator training
# --------------------------------------

def train(root_folder, hyp_params):
    folder_struct = FolderStructureBuilder(root_folder, ModelTypes.ROI)
    # get folder names for saving
    _, params_pickle, measur_pickle, visuals, weights = folder_struct.get_training_paths()

    # gather the parameters
    data_folder = hyp_params.data_folder
    batch_size = hyp_params.batch_size
    val_ratio = hyp_params.val_ratio
    epochs = hyp_params.epochs
    lr = hyp_params.lr
    device = hyp_params.device
    criterion = hyp_params.criterion

    loader = ROITrainLoader(data_folder, device, batch_size, val_ratio)
    loaders = create_loaders(loader)

    model = ResNet18roi().to(device)
    opt = optim.Adam(model.parameters(), lr, (0.9, 0.98), weight_decay=0)
    hyp_params.optimizer = opt
    hyp_params.criterion = None

    # save the parameters
    hyp_params.serialize(params_pickle)

    # instantiating a result object, file_name refers to measur_pickle
    results = Results(measur_pickle, visuals, weights)

    training_algo = One2OneOptimizer(model, opt, criterion, loaders, epochs, results, None)
    training_algo.fit()


if __name__ == '__main__':
    results_folder = config.roi_result_folder

    hyp_params = Parameters()  
    hyp_params.data_folder = config.roi_data
    hyp_params.batch_size = 8
    hyp_params.val_ratio = 0.1
    hyp_params.epochs = 70
    hyp_params.lr = 1e-3
    hyp_params.device = torch.device('cuda')  
    hyp_params.criterion = nn.MSELoss()

    train(results_folder, hyp_params)
