from dataloader_utils import FolderStructureBuilder, ModelTypes, Visualizer, Side
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from datatransform_utils import ControlPoints
from training_utils import One2OneOptimizer
from training_utils import ConGradient
from dataloader_utils import HeartPart
import torchvision.models as models
from utils import progress_bar
from preproc import AugmentROI
from metrics import dice
from torch import optim
import torch.nn as nn
import numpy as np
import config
import pickle
import torch
import os


class ResNet34cp(nn.Module):

    def __init__(self, controls):
        super(ResNet34cp, self).__init__()
        self.controls = controls  # number of control points on a curve

        # creating the network architecture
        self.conv = nn.Conv2d(1, 3, (1, 1), 1)
        self.resnet = models.resnet34(num_classes=512)

        self.linear = nn.Linear(512, self.controls * 2)
        self.prelu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        temp = self.prelu(self.conv(x))
        temp = self.prelu(self.resnet(temp))
        temp = self.sigmoid(self.linear(temp))
        return temp.view(-1, self.controls, 2)

    def save(self, path):
        torch.save(self, path)

    @classmethod
    def load(cls, controls, state_dict, device):
        model = cls(controls)
        model.load_state_dict(state_dict)
        return model.to(device)


class ControlPointsTrainLoader:

    def __init__(self, folder, device, batch_size=32, validation_ratio=0.05):
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
        self.train_weights, self.train_imgs, self.train_points_left, self.train_points_right, image_id = self._unfold_samples(self.train_samples, 0)
        self.val_weights, self.val_imgs, self.val_points_left, self.val_points_right, _ = self._unfold_samples(self.val_samples, image_id)

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

    def _unfold_samples(self, samples, start_id):
        weights = []
        images = []
        points_left = []
        points_right = []
        image_id = 0
        for sample in samples:
            patient_id = sample.patient_id
            slices = sorted(list(sample.image_contour_3s.keys()))
            lower_bound = len(slices) // 3
            upper_bound = len(slices) // 3 * 2
            for slice in sample.image_contour_3s:
                added_image_phases = []
                for part in sample.image_contour_3s[slice]:
                    for phase in sample.image_contour_3s[slice][part]:
                        descr = (patient_id, slice, part, phase)
                        if part == HeartPart.LN:
                            if not(phase in added_image_phases):
                                images.append(sample.image_contour_3s[slice][part][phase]['image'])
                                weights.append(1.0 if lower_bound < slice < upper_bound else 1.2)
                                points_left.append(sample.image_contour_3s[slice][part][phase]['mask'])
                                points_right.append(None)
                                added_image_phases.append(phase)
                                self.hash_original_predicted[image_id + start_id] = descr
                                image_id += 1
                            else:
                                idx = image_id - 2 + added_image_phases.index(phase)
                                points_left[idx] = sample.image_contour_3s[slice][part][phase]['mask']
                        if part == HeartPart.RN:
                            if not(phase in added_image_phases):
                                images.append(sample.image_contour_3s[slice][part][phase]['image'])
                                weights.append(1.0 if lower_bound < slice < upper_bound else 1.2)
                                points_left.append(None)
                                points_right.append(sample.image_contour_3s[slice][part][phase]['mask'])
                                added_image_phases.append(phase)
                                self.hash_original_predicted[image_id + start_id] = descr
                                image_id += 1
                            else:
                                idx = image_id - 2 + added_image_phases.index(phase)
                                points_right[idx] = sample.image_contour_3s[slice][part][phase]['mask']
        return weights, images, points_left, points_right, image_id

    # Inner class for wrapping the data in a dataset
    class ControlDataset(Dataset):
        """
        Dataset for wrapping the raw images and the corresponding binaries.
        This makes possible to use DataLoader which has several useful properties.
        """

        def __init__(self, device, batch_size, images, controls):
            """
            device - torch.device, gpu or cpu
            images - list of all the images
            masks - a tuple with masks (left, right), one of them can be None
            """
            self.device = device
            self.images = images
            self.points_left = controls[0]
            self.points_right = controls[1]
            self.index_mapping()
            self.augment = AugmentROI((140, 140), (-10, 11), (-15, 16), 0.001).preprocess
        
        def add_points(self, contour):
            cp_num = 40
            N = contour.shape[0]
            if N < cp_num:
                ext_contour = np.zeros((cp_num, 2), dtype=np.float32)
                ext_contour[:N, :] = contour
                for idx in range(N, cp_num):
                    ext_contour[idx, :] = contour[N-1, :]
                contour = ext_contour
            return contour
        
        def index_mapping(self):
            if not(self.points_left is None):
                self.left_idcs = [idx for idx in range(len(self.images)) if self.points_left[idx] is not None]
            if not(self.points_right is None):
                self.right_idcs = [idx for idx in range(len(self.images)) if self.points_right[idx] is not None]

        def __len__(self):
            return len(self.left_idcs) if self.points_left is not None else len(self.right_idcs)

        def __getitem__(self, index):
            if not(self.points_left is None):
                idx = self.left_idcs[index]
                img, contours = self.augment(self.images[idx], [self.points_left[idx]])
                contour = self.add_points(contours[0])
                sample = {
                    'image_id': idx,
                    'image': torch.tensor(img, dtype=torch.float, device=self.device).unsqueeze(0),
                    'target': torch.tensor(contour, dtype=torch.float, device=self.device)
                }
            elif not(self.points_right is None):
                idx = self.right_idcs[index]
                #img, contours, _ = self.ccrop(self.images[idx], [self.points_right[idx]])
                img, contours = self.augment(self.images[idx], [self.points_right[idx]])
                #img, contours, _ = self.scale(img, contours, 1.0, (224, 224))
                contour = self.add_points(contours[0])
                sample = {
                    'image_id': idx,
                    'image': torch.tensor(img, dtype=torch.float, device=self.device).unsqueeze(0),
                    'target': torch.tensor(contour, dtype=torch.float, device=self.device)
                }
            else:
                sample = None

            return sample

    def get_left_trainloader(self):
        """
        Creates a data loader to iterate through the left training part of the dataset.
        """
        dataset = self.ControlDataset(self.device, self.batch_size, self.train_imgs, (self.train_points_left, None))
        left_train_weights = [self.train_weights[idx] for idx in range(len(self.train_weights)) if self.train_points_left[idx] is not None]
        weights = torch.DoubleTensor(left_train_weights)
        sampler = WeightedRandomSampler(weights, len(weights))
        return DataLoader(dataset, sampler=sampler, batch_size=self.batch_size, shuffle=False)

    def get_left_validationloader(self):
        """
        Creates a data loader to iterate through left the test part of the dataset.
        """
        dataset = self.ControlDataset(self.device, self.batch_size, self.val_imgs, (self.val_points_left, None))
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def get_right_trainloader(self):
        """
        Creates a data loader to iterate through the right training part of the dataset.
        """
        dataset = self.ControlDataset(self.device, self.batch_size, self.train_imgs, (None, self.train_points_right))
        right_train_weights = [self.train_weights[idx] for idx in range(len(self.train_weights)) if self.train_points_right[idx] is not None]
        weights = torch.DoubleTensor(right_train_weights)
        sampler = WeightedRandomSampler(weights, len(weights))
        return DataLoader(dataset, sampler=sampler, batch_size=self.batch_size, shuffle=False)

    def get_right_validationloader(self):
        """
        Creates a data loader to iterate through the right test part of the dataset.
        """
        dataset = self.ControlDataset(self.device, self.batch_size, self.val_imgs, (None, self.val_points_right))
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)


class Parameters:

    def __init__(self):
        self.data_folder = ""
        self.batch_size = None
        self.val_ratio = None
        self.epochs = None
        self.lr = None
        self.side = None
        self.device = None
        self.control_num = None
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
            trg = ControlPoints.controls2curve(target[idx])
            prd = ControlPoints.controls2curve(predicted[idx])
            dice_value= dice(trg, prd)
            self.c_val_pred.append(dice_value)

    def cache_train_loss(self, loss):
        self.c_train_loss.append(loss)
        if len(self.c_train_loss) > 5:
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
            pass
            h, w = self.images[0][0].shape
            summary_image = np.zeros((1, 3 * h, len(self.images) * w))
            for col, tercier in enumerate(self.images):
                tercier = list(tercier)
                tercier[1] = ControlPoints.controls2image((h, w), tercier[1])
                tercier[2] = ControlPoints.controls2image((h, w), tercier[2])
                for row, img in enumerate(tercier):
                    summary_image[0, row * h: (row + 1) * h, col * w: (col + 1) * w] = img
            self.vszr.visualize_image(summary_image, global_counter)
            self.images.clear()


def create_loaders(loader, side):
    if side == Side.RIGHT:
        return loader.get_right_trainloader(), loader.get_right_validationloader()
    elif side == Side.LEFT:
        return loader.get_left_trainloader(), loader.get_left_validationloader()
    else:
        return None


def train(root_folder, hyp_params):
    folder_struct = FolderStructureBuilder(root_folder, ModelTypes.CP)
    # get folder names for saving
    _, params_pickle, measur_pickle, visuals, weights = folder_struct.get_training_paths()

    # gather the parameters
    data_folder = hyp_params.data_folder
    batch_size = hyp_params.batch_size
    val_ratio = hyp_params.val_ratio
    epochs = hyp_params.epochs
    lr = hyp_params.lr
    side = hyp_params.side
    device = hyp_params.device
    control_num = hyp_params.control_num
    criterion = hyp_params.criterion

    loader = ControlPointsTrainLoader(data_folder, device, batch_size, val_ratio)
    loaders = create_loaders(loader, side)

    model = ResNet34cp(control_num).to(device)
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
    results_folder = config.cp_result_folder

    hyp_params = Parameters()
    hyp_params.data_folder = config.cp_data
    hyp_params.batch_size = 8
    hyp_params.val_ratio = 0.1
    hyp_params.epochs = 70
    hyp_params.lr = 1e-3
    hyp_params.side = Side.LEFT if config.side == 'left' else Side.RIGHT
    hyp_params.control_num = 40
    hyp_params.device = torch.device('cuda')
    hyp_params.criterion = ConGradient()

    train(results_folder, hyp_params)
