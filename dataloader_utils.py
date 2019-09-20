from tensorboardX import SummaryWriter
from metrics import dice
from enum import Enum
import numpy as np
import datetime
import pickle
import random
import os


class Sample:

    def __init__(self):
        self.patient_id = None
        self.gender = None
        self.volume_indices = None      # the volume indices calc. from the ground truth contours
        self.ratio = None               # ratio between pixel number and real volume
        self.bsa = None                 # body surface area
        self.image_contour_3s = None    # hierarchically ordered image, left-, right masks, left-, right pred. (slice, part, phase)


class Side(Enum):
    BOTH = 1
    LEFT = 2
    RIGHT = 3


class Gender(Enum):
    X = 0  # unknown
    M = 1
    F = 2


class EndPhase(Enum):
    UNK = 0  # unknown
    SYS = 1  # systole
    DIA = 2  # diastole


class HeartPart(Enum):
    UNK = 0  # unknown
    LN = 1  # left-endo
    LP = 2  # left-epi
    RN = 3  # right-endo
    RP = 4  # right-epi

    
class ModelTypes(Enum):
    AE = 1   # auro-encoder for scare segmentation
    BIN = 2  # binary mask for slice-wise segmentation (SA)
    CP = 3   # control points based segmentation (SA)
    ROI = 4  # models for finding the roi on a heart image
    GAN = 5
    MYO = 6
    LVQ = 7  # Stacom2019 challenge


class Visualizer:

    def __init__(self, root_folder):
        self.root_folder = root_folder
        self.writer = SummaryWriter(root_folder)
        self.scalar_board = 'errors/'

    def visualize_loss(self, losses, counter):
        data_point = {'training': losses[0], 'validation': losses[1]}
        self.writer.add_scalars(self.scalar_board + 'learning_curve', data_point, counter)

    def visualize_dices(self, dices, counter):
        data_point = {'training': dices[0], 'validation': dices[1]}
        self.writer.add_scalars(self.scalar_board + 'dice_value', data_point, counter)

    def visualize_indices(self, name, index_rel_diffs):
        index_rel_diffs = np.array(index_rel_diffs)
        self.writer.add_histogram('volume_indices/' + name, index_rel_diffs, 0)

    def visualize_hystogram(self, name, data):
        data = np.array(data)
        self.writer.add_histogram(name, data, 0)

    def visualize_weights(self, name, weights, counter):
        if name.find('deconv') != -1:
            group_name = 'Deconvolutions/' + name
        elif name.find('conv') != -1:
            group_name = 'Convolutions/' + name
        elif name.find('batch') != -1:
            group_name = 'BatchNorms/' + name
        else:
            group_name = 'Other/' + name
        self.writer.add_histogram('weights_' + group_name, weights, counter)

    def visualize_network(self, model, input_data):
        self.writer.add_graph(model, input_data)

    def visualize_image(self, image, counter):
        self.writer.add_image('ValidationImages', image, counter)

    def __del__(self):
        self.writer.close()


class FolderStructureBuilder:

    def __init__(self, root_folder, model_type):
        self.root_folder = root_folder
        self.model_type = str(model_type)
        self.folder_uuid = {
            'train': self._get_uuid('train'),
            'eval': self._get_uuid('eval')
        }

        # creating: root folder if necessary
        self._safe_mkdir(root_folder)

        # creating a folder corresponding to the given model if necessary
        self.model_folder = os.path.join(root_folder, self.model_type)
        self._safe_mkdir(self.model_folder)

        # creating the training folder if necessary
        self.training_folder = os.path.join(self.model_folder, 'trainings')
        self._safe_mkdir(self.training_folder)

        # creating the evaluation folder if necessary
        self.eval_folder = os.path.join(self.model_folder, 'evals')
        self._safe_mkdir(self.eval_folder)

    def get_training_paths(self):
        folder_uuid = self.folder_uuid['train']
        # creating new folder for the new experiment
        exp_folder = os.path.join(self.training_folder, folder_uuid)
        self._safe_mkdir(exp_folder)

        # creating names for parameters and measurements objects
        params_pickle = os.path.join(exp_folder, 'params.pickle')
        measur_pickle = os.path.join(exp_folder, 'measur.pickle')

        # creating the visualization folder
        visuals = os.path.join(exp_folder, 'visuals')

        # creating the checkpoint folder
        checkpoints = os.path.join(exp_folder, 'checkpoints')
        self._safe_mkdir(checkpoints)

        return folder_uuid, params_pickle, measur_pickle, visuals, checkpoints

    def get_evaluation_paths(self):
        folder_uuid = self.folder_uuid['eval']
        # creating new folder for the new experiment
        exp_folder = os.path.join(self.eval_folder, folder_uuid)
        self._safe_mkdir(exp_folder)

        # creating the visualization folder
        visuals = os.path.join(exp_folder, 'visuals')

        # creating names for measurements and meta files
        measur_pickle = os.path.join(exp_folder, 'measur.pickle')
        meta_json = os.path.join(exp_folder, 'meta.json')

        return folder_uuid, measur_pickle, meta_json, visuals

    def _get_uuid(self, preamble):
        uuid = preamble
        uuid = uuid + '_' + (str(datetime.datetime.now()).
                             replace('-', '').
                             replace(' ', '').
                             replace(':', '').
                             replace('.', '')
                             )
        uuid = uuid + '_' + str(random.randint(1, 1000))
        return uuid

    @staticmethod
    def _safe_mkdir(path):
        if not(os.path.exists(path)):
            os.mkdir(path)
