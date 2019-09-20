from dataloader_utils import FolderStructureBuilder, ModelTypes, Side
from preproc import CenterCropnoScale, CropROIfromCenter, Proc
from dataloader_utils import HeartPart, EndPhase
from datatransform_utils import ControlPoints
from train_roi import ResNet18roi
from train_cp import ResNet34cp
from utils import progress_bar
from train_cp import Results
from metrics import *
from enum import Enum
import numpy as np
import pickle
import shutil
import config
import torch
import json
import os


def evaluation(root_folder, test_data, device, others, area_of_predicted, calculate_similarities, make_prediction, vs):
    # -------------------------------------
    # defining the HELPER functions
    #   
    def convert_to_contour_areas(hierarchical):
        contour_areas = {}
        for slice in hierarchical:
            if not(slice in contour_areas):
                contour_areas[slice] = {}
            for part in [HeartPart.LN, HeartPart.RN]:
                if not(part in contour_areas[slice]):
                    contour_areas[slice][part] = {}
                for phase in [EndPhase.DIA, EndPhase.SYS]:
                    contour_areas[slice][part][phase] = None
                    if part in hierarchical[slice]:
                        if phase in hierarchical[slice][part]:
                            pred_mask = hierarchical[slice][part][phase]['pred_mask']
                            ratio = hierarchical[slice][part][phase]['ratio']
                            if not(pred_mask is None):
                                contour_areas[slice][part][phase] = area_of_predicted(pred_mask) * ratio
        return contour_areas
        
    def similarity(hierarchical, mm_ratio):
        dcs = []  # dice values
        jcs = []  # jaccard values
        hfs = []  # hausdorff values
        for slice in hierarchical:
            for part in hierarchical[slice]:
                for phase in hierarchical[slice][part]:
                    if not(hierarchical[slice][part][phase]['pred_mask'] is None):
                        dc, jc, hf = calculate_similarities(hierarchical[slice][part][phase])
                        dcs.append((phase, part, dc))
                        jcs.append((phase, part, jc))
                        if not(hf is None):
                            hfs.append((phase, part, hf * mm_ratio))
        return dcs, jcs, hfs

    def save_images(name, image, original, predicted):
        r = np.random.randint(1, 101)
        if r == 1:
            folder = os.path.join(vs.vszr.root_folder, 'images')
            if not os.path.exists(folder):
                os.mkdir(folder)
            name = os.path.join(folder, name)
            np.save(name + '_image.npy', image, allow_pickle=False)
            np.save(name + '_orig.npy', original, allow_pickle=False)
            np.save(name + '_pred.npy', predicted, allow_pickle=False)

    # initialize the model
    side = others
        
    # storing the calculated metrics
    volumes = []
    vindices = []
    dices = []
    jaccards = []
    hausdorffs = []

    # -------------------------------------
    # evaluation on the test samples
    #
    samples_as_pickle = os.listdir(test_data)
    for cntr, sample_as_pickle in enumerate(samples_as_pickle, 1):
        path = os.path.join(test_data, sample_as_pickle)
        with open(path, 'br') as f:
            sample = pickle.load(f)
        # prediction
        part = HeartPart.LN if side == Side.LEFT else HeartPart.RN
        for slice in sample.image_contour_3s:
            if part in sample.image_contour_3s[slice]:
                for phase in sample.image_contour_3s[slice][part]:
                    image = sample.image_contour_3s[slice][part][phase]['image']
                    contour = sample.image_contour_3s[slice][part][phase]['mask']
                    predicted, image, orig, ratio = make_prediction(image, [contour])
                    sample.image_contour_3s[slice][part][phase]['pred_mask'] = predicted
                    sample.image_contour_3s[slice][part][phase]['mask'] = orig
                    sample.image_contour_3s[slice][part][phase]['ratio'] = ratio
                    save_images(str(cntr), image * 255, orig, predicted)

        # volume indices
        hierarchical = convert_to_contour_areas(sample.image_contour_3s)
        if side == Side.LEFT:
            volume_left = calculate_volumes_left(hierarchical, sample.ratio, sample.bsa)
            volume_indices = {'predicted': volume_left, 'baseline': sample.volume_indices}
        elif side == Side.RIGHT:
            volume_right = calculate_volumes_right(hierarchical, sample.ratio, sample.bsa)
            volume_indices = {'predicted': volume_right, 'baseline': sample.volume_indices}
        else:
            volume_left = calculate_volumes_left(hierarchical, sample.ratio, sample.bsa)
            volume_right = calculate_volumes_right(hierarchical, sample.ratio, sample.bsa)
            volume_indices = {'predicted': {**volume_left, **volume_right}, 'baseline': sample.volume_indices}
        volumes.append(volume_indices)  
        o = VolumeIndices.from_dictionary(volume_indices['baseline'], sample.gender)
        p = VolumeIndices.from_dictionary(volume_indices['predicted'], sample.gender)
        vindices.append((o, p))  # reordering calculation requires VolumeIndices objects

        # dice, jaccard, hausdorff
        dc, jc, hf = similarity(sample.image_contour_3s, np.sqrt(sample.ratio[0] * 1000 / 8))
        dices.append(dc)
        jaccards.append(jc)
        hausdorffs.append(hf)

        progress_bar(cntr, len(samples_as_pickle), 10)

    # -------------------------------------
    # reporting results
    #

    # build hystogram arrays
    similarity_dict = {
        'dice': {
            EndPhase.SYS: {
                HeartPart.LN: [],
                HeartPart.RN: []
            },
            EndPhase.DIA: {
                HeartPart.LN: [],
                HeartPart.RN: []
            }
        },
        'jaccard': {
            EndPhase.SYS: {
                HeartPart.LN: [],
                HeartPart.RN: []
            },
            EndPhase.DIA: {
                HeartPart.LN: [],
                HeartPart.RN: []
            }
        },
        'hausdorff': {
            EndPhase.SYS: {
                HeartPart.LN: [],
                HeartPart.RN: []
            },
            EndPhase.DIA: {
                HeartPart.LN: [],
                HeartPart.RN: []
            }
        },
    }
    for dcs, jcs, hfs in zip(dices, jaccards, hausdorffs):
        for dc in dcs:
            similarity_dict['dice'][dc[0]][dc[1]].append(dc[2])
            similarity_dict['dice'][dc[0]][dc[1]].append(dc[2])
        for jc in jcs:
            similarity_dict['jaccard'][jc[0]][jc[1]].append(jc[2])
            similarity_dict['jaccard'][jc[0]][jc[1]].append(jc[2])
        for hf in hfs:
            similarity_dict['hausdorff'][hf[0]][hf[1]].append(hf[2])
            similarity_dict['hausdorff'][hf[0]][hf[1]].append(hf[2])
    
    with open(os.path.join(vs.vszr.root_folder, 'similarity_metrics.json'), 'w') as f:
        json.dump(str(similarity_dict), f)

    for mc in ['dice', 'jaccard', 'hausdorff']:
        for phase in [EndPhase.SYS, EndPhase.DIA]:
            for side in [HeartPart.LN, HeartPart.RN]:
                if len(similarity_dict[mc][phase][side]) > 0:
                    vs.visualize_hystogram(mc + '/' + phase.name + '_' + side.name, similarity_dict[mc][phase][side])

    # calculating volume index deviations in percentage (ratio)
    rel_volume_indices = {}
    for idc in ['lved', 'lves', 'lvsv', 'lved_i', 'lves_i', 'lvsv_i', 'rved', 'rves', 'rvsv', 'rved_i', 'rves_i', 'rvsv_i']:
        rel_idcs = []
        for vol_idcs in volumes:
            if idc in vol_idcs['predicted']:
                rel_idc = abs((vol_idcs['predicted'][idc] - vol_idcs['baseline'][idc]) / (vol_idcs['baseline'][idc] + 1e-5))
                rel_idcs.append(rel_idc)
        if len(rel_idcs) > 0:
            vs.visualize_hystogram('volume_indices/' + idc, rel_idcs)
            rel_volume_indices[idc] = rel_idcs
    
    with open(os.path.join(vs.vszr.root_folder, 'rel_volume_indices.json'), 'w') as f:
        json.dump(str(rel_volume_indices), f)
    
    # calculating volume index deviations in percentage (ratio)
    abs_volume_indices = {}
    for idc in ['lved', 'lves', 'lvsv', 'lved_i', 'lves_i', 'lvsv_i', 'rved', 'rves', 'rvsv', 'rved_i', 'rves_i', 'rvsv_i']:
        abs_idcs = []
        for vol_idcs in volumes:
            if idc in vol_idcs['predicted']:
                abs_idc = abs((vol_idcs['predicted'][idc] - vol_idcs['baseline'][idc]))
                abs_idcs.append(abs_idc)
        if len(abs_idcs) > 0:
            vs.visualize_hystogram('volume_indices/' + idc, abs_idcs)
            abs_volume_indices[idc] = abs_idcs
    
    with open(os.path.join(vs.vszr.root_folder, 'abs_volume_indices.json'), 'w') as f:
        json.dump(str(abs_volume_indices), f)
    
    # calculating reordering values
    rp = ReorderPercentage(vindices)
    all_errors, ln, nh, nl, hn, lh, hl = rp.reordering_percentage()

    with open(os.path.join(vs.vszr.root_folder, 'reorder_percentages.json'), 'w') as f:
        reorder_dicts = {'all_errors': all_errors, 'ln': ln, 'nh': nh, 'nl': nl, 'hn': hn, 'lh': lh, 'hl': hl}
        json.dump(str(reorder_dicts), f)

results_folder = config.evaluation_result_folder
test_data = config.evaluation_data
checkpoint_path_cp = config.cp_right_weights if config.side == 'right' else config.cp_left_weights
checkpoint_path_roi = config.roi_right_weights if config.side == 'right' else config.roi_left_weights
device = torch.device('cuda')

side, control_num = (Side.RIGHT if config.side == 'right' else Side.LEFT, 40)
def area_of_predicted(pred_mask):
    contour = ControlPoints.controls2curve(pred_mask / 224) * 224
    return calculate_contour_area(contour)
            
def calculate_similarities(hierarchical):
    pred = ControlPoints.controls2curve(hierarchical['pred_mask'] / 224)
    orig = ControlPoints.controls2curve(hierarchical['mask'] / 224)
    dc = dice(orig.copy(), pred.copy())
    jc = jaccard(orig.copy(), pred.copy())
    hf = hausdorff(orig.copy(), pred.copy()) * 224 * np.sqrt(hierarchical['ratio'])
    return dc, jc, hf

if __name__ == '__main__':
    ccrop = CenterCropnoScale((224, 224))
    # load the ROI model
    checkpoint = torch.load(checkpoint_path_roi)
    model_state = checkpoint["model_state"]
    model_roi = ResNet18roi.load(model_state, device)
    # load the CP model
    checkpoint = torch.load(checkpoint_path_cp)
    model_state = checkpoint["model_state"]
    model_cp = ResNet34cp.load(control_num, model_state, device)
    def make_prediction(image, contours):
        image, contours, ratio1 = ccrop.preprocess(image, contours)
        size = image.shape[0]
        image2 = torch.tensor(image.copy(), dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
        center = model_roi(image2).cpu().detach().numpy()[0] * size
        crop_roi = CropROIfromCenter((224, 224), (140, 140), (int(center[1]), int(center[0])))
        image, contours, ratio2 = crop_roi.preprocess(image, contours)
        image3 = torch.tensor(image.copy(), dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
        size = image.shape[0]
        predicted = model_cp(image3)
        predicted = predicted.cpu().detach().numpy()[0] * size
        image = image3.cpu().detach().numpy()[0, 0]
        return predicted, image, contours[0], ratio1 * ratio2

    folder_struct = FolderStructureBuilder(results_folder, ModelTypes.CP)
    _, _, meta_json, visuals = folder_struct.get_evaluation_paths()
    vs = Results(None, visuals, None)
                
    # calling the common procedures to calculate the values of interest
    evaluation(results_folder, test_data, device, side, area_of_predicted, calculate_similarities, make_prediction, vs)
