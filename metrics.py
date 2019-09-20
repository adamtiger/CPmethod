from dataloader_utils import Gender, HeartPart, EndPhase
from enum import Enum
import numpy as np
import math
import cv2


# --------------------------------------
# Shape (contour) similarity
# --------------------------------------

def __areas(curve1, curve2):
    # floats come in
    # find the corners of the bbox
    def _bbox(cv):
        mins = np.min(cv, axis=0)
        maxs = np.max(cv, axis=0)
        x_min, y_min = mins[0], mins[1]
        x_max, y_max = maxs[0], maxs[1]
        return x_min, y_min, x_max, y_max

    box1 = _bbox(curve1)
    box2 = _bbox(curve2)
    xr = max(box1[2], box2[2])
    yb = max(box1[3], box2[3])
    xl = min(box1[0], box2[0])
    yu = max(box1[1], box2[1])

    # shift and rescale the curves (DC, JC will not change)
    curve1[:, 0] = (curve1[:, 0] - xl) / (xr - xl + 1e-5)
    curve1[:, 1] = (curve1[:, 1] - yu) / (yb - yu + 1e-5)
    curve2[:, 0] = (curve2[:, 0] - xl) / (xr - xl + 1e-5)
    curve2[:, 1] = (curve2[:, 1] - yu) / (yb - yu + 1e-5)

    # map the coordinates to 410 x 410 mask
    image1 = np.zeros((410, 410), dtype=np.uint8)
    curve1 = curve1 * 400 + 5
    cv2.drawContours(image1, [np.expand_dims(curve1, axis=1).astype(np.int32)], -1, (255, 0, 0), cv2.FILLED)

    image2 = np.zeros((410, 410), dtype=np.uint8)
    curve2 = curve2 * 400 + 5
    cv2.drawContours(image2, [np.expand_dims(curve2, axis=1).astype(np.int32)], -1, (255, 0, 0), cv2.FILLED)

    A = (image1 // 255 == 1).astype(np.float32)
    B = (image2 // 255 == 1).astype(np.float32)

    area1 = np.sum(A)
    area2 = np.sum(B)
    area_inter = np.sum(A * B)
    area_union = area1 + area2 - area_inter
    return area_union, area_inter, area1, area2


def dice(curve1, curve2):  # can be viewed as F1 score
    """
    Calculate the dice metric for the two curves.
    :param curve1: a numpy matrix with shape (N, 2), points are in x, y format
            elements are integers
    :param curve2: a numpy matrix with shape (N, 2), points are in x, y format
            elements are integers
    :return: a real number (the dice value)
    """
    _, inter, a1, a2 = __areas(curve1, curve2)

    # dice metric
    return 2.0 * inter / (a1 + a2)


def jaccard(curve1, curve2):  # aka. Tanimoto index
    """
    Calculate the jaccard metric for the two curves.
    :param curve1: a numpy matrix with shape (N, 2), points are in x, y format
            elements are integers
    :param curve2: a numpy matrix with shape (N, 2), points are in x, y format
            elements are integers
    :return: a real number (the jaccard index)
    """
    union, inter, _, _ = __areas(curve1, curve2)

    # dice metric
    return inter / union


def hausdorff(curve1, curve2):  # aka. Pompeiu-Hausdorff distance
    """
    Calculate the Hausdorff distance between two curves. (https://en.wikipedia.org/wiki/Hausdorff_distance)
    :param curve1: a numpy matrix with shape (N, 2), points are in x, y format
    :param curve2: a numpy matrix with shape (N, 2), points are in x, y format
    :return: a real number (hausdorff distance)
    """
    N2 = curve2.shape[0]
    temp = np.expand_dims(curve1, 2)
    temp = np.repeat(temp, N2, 2)
    temp = temp - curve2.T
    distances = temp[:, 0, :] ** 2 + temp[:, 1, :] ** 2
    d1 = np.max(np.min(distances, 0))
    d2 = np.max(np.min(distances, 1))
    return math.sqrt(max(d1, d2))


# --------------------------------------
# Volume calculation
# --------------------------------------

def ratio(pixel_spacing: tuple, slice_thickness: float, gap: float) -> (float, float):
    ratio_slice = pixel_spacing[0] * pixel_spacing[1] * slice_thickness / 1000.0  # mm^3 -> ml conversion
    ratio_gap = pixel_spacing[0] * pixel_spacing[1] * gap / 1000.0
    return ratio_slice, ratio_gap


def bsa(height, weight):  # Mosteller BSA
    if not(height is None or weight is None):
        return math.sqrt(height * weight / 3600.0)
    else:
        return None

    
def area_triangular(curve):
    """
    Calculates the area of a closed curve based on
    crossproducts.
    :param curve: a numpy matrix with shape (N, 2), points are in x, y format
           elements are floats
    :return: area
    """
    # calculate center of mass
    crm = np.sum(curve, axis=0) / curve.shape[0]

    # vector between crm and a point of the curve
    r = curve - crm

    # side vector
    curve_mtx_shifted = np.ones_like(curve)
    curve_mtx_shifted[0] = curve[-1]
    curve_mtx_shifted[1:] = curve[0:-1]
    dr = curve - curve_mtx_shifted

    # vector product
    rxdr = np.cross(r, dr)

    # sum up the pieces of triangulars
    return np.abs(0.5 * np.sum(rxdr))
    

def convert_to_hierarchical(contours):
    """
    convert list of contours into a hierarchical structure
    slice > frame > heartpart -- Contour
    :param contours: list of Contour objects
    :return: a hierarchical structure which contains Contour objects
    """
    hierarchical_contours = {}
    for contour in contours:
        if not(contour.slice in hierarchical_contours.keys()):
            hierarchical_contours[contour.slice] = {}
        if not(contour.frame in hierarchical_contours[contour.slice].keys()):
            hierarchical_contours[contour.slice][contour.frame] = {}
        hierarchical_contours[contour.slice][contour.frame][contour.part] = contour
    return hierarchical_contours


def calculate_contour_area(curve: np.ndarray):
    """
    calculate area with triangulars
    :param curve: numpy matrix (N, 2)
    :return: area of the closed curve
    """
    return area_triangular(curve)


def grouping(hierarchical_contours, calculate_area):
    """
    Determines the contour which phase belongs to (systole or diastole).
    Calculates the areas of each contour.
    :param hierarchical_contours: a hierarchical structure which contains Contour objects
    (slice > frame > heartpart -- Contour)
    :param calculate_area: function to calculate area of the contour
    :return: hierarchical structure with areas (slice > heartpart > phase -- area)
    """
    def set_endphase(slice, frame, part, phase):
        hierarchical_contours[slice][frame][part].phase = phase
        hierarchical_contours[slice][frame][part].corresponding_image.phase = phase

    contour_areas = {}
    slices = hierarchical_contours.keys()
    for slice in slices:
        contour_areas[slice] = {}
        for part in HeartPart:
            areas = []
            frames = []
            contour_areas[slice][part] = {}
            for frame in hierarchical_contours[slice].keys():
                if part in hierarchical_contours[slice][frame]:
                    curve = hierarchical_contours[slice][frame][part]
                    frames.append(frame)
                    areas.append(calculate_area(curve.contour_mtx))

            if len(areas) > 1:
                contour_areas[slice][part][EndPhase.DIA] = max(areas)
                contour_areas[slice][part][EndPhase.SYS] = min(areas)
                set_endphase(slice, frames[areas.index(max(areas))], part, EndPhase.DIA)
                set_endphase(slice, frames[areas.index(min(areas))], part, EndPhase.SYS)
            elif len(areas) == 1:
                ds = np.array([frames[0] - 0, frames[0] - 20, frames[0] - 9])  # this is a heuristic
                idx = np.argmin(np.abs(ds))
                if idx in [0, 1]:
                    contour_areas[slice][part][EndPhase.DIA] = areas[0]
                    contour_areas[slice][part][EndPhase.SYS] = None
                    set_endphase(slice, frames[0], part, EndPhase.DIA)
                else:
                    contour_areas[slice][part][EndPhase.DIA] = None
                    contour_areas[slice][part][EndPhase.SYS] = areas[0]
                    set_endphase(slice, frames[0], part, EndPhase.SYS)
            else:
                contour_areas[slice][part][EndPhase.DIA] = None
                contour_areas[slice][part][EndPhase.SYS] = None
    return contour_areas


def volume(contour_areas, part, phase, ratio):
    """
    :param contour_areas: hierarchical structure with areas (slice > heartpart > phase -- area)
    :param part: heartpart e.g.: left-endo
    :param phase: systole or diastole
    :param ratio: comes from the field view, volume changing and slice thickness
    :return: volume of the heart in part at phase
    """
    ratio_slice, ratio_gap = ratio

    v = 0
    slices = list(contour_areas.keys())
    for idx in range(len(slices) - 1):
        a1 = contour_areas[slices[idx]][part][phase]
        a2 = contour_areas[slices[idx + 1]][part][phase]
        if a1 is not None:
            v += a1 * ratio_slice
            if a2 is not None:
                v += (a1 + np.sqrt(a1 * a2) + a2) * ratio_gap / 3.0
    a1 = contour_areas[slices[-1]][part][phase]  # the last slice
    if a1 is not None:
        v += a1 * ratio_slice

    return v


def calculate_volumes_left(contour_areas, ratio, bsa=None):
    lved = volume(contour_areas, HeartPart.LN, EndPhase.DIA, ratio)  # left ED
    lves = volume(contour_areas, HeartPart.LN, EndPhase.SYS, ratio)  # left ES
    lvsv = lved - lves                     # left Stroke-volume

    volume_indices = {'lved': lved, 'lves': lves, 'lvsv': lvsv}

    # other metrics: left
    if bsa is None:
        return volume_indices

    lved_i = lved / bsa                      # left ED-index
    lves_i = lves / bsa                      # left ES-index
    lvsv_i = lvsv / bsa                      # left SV-index

    volume_indices['lved_i'] = lved_i
    volume_indices['lves_i'] = lves_i
    volume_indices['lvsv_i'] = lvsv_i
    return volume_indices


def calculate_volumes_right(contour_areas, ratio, bsa=None):
    rved = volume(contour_areas, HeartPart.RN, EndPhase.DIA, ratio)
    rves = volume(contour_areas, HeartPart.RN, EndPhase.SYS, ratio)
    rvsv = rved - rves  # right Stroke-volume

    volume_indices = {'rved': rved, 'rves': rves, 'rvsv': rvsv}

    # other metrics: right
    if bsa is None:
        return volume_indices

    rved_i = rved / bsa                      # right ED-index
    rves_i = rves / bsa                      # right ES-index
    rvsv_i = rvsv / bsa                      # right SV-index

    volume_indices['rved_i'] = rved_i
    volume_indices['rves_i'] = rves_i
    volume_indices['rvsv_i'] = rvsv_i
    return volume_indices


class VolumeIndices:

    def __init__(self):
        self.gender = None

        self.lved = None
        self.lves = None
        self.lvsv = None
        self.lved_i = None
        self.lves_i = None
        self.lvsv_i = None

        self.rved = None
        self.rves = None
        self.rvsv = None
        self.rved_i = None
        self.rves_i = None
        self.rvsv_i = None

    @classmethod
    def from_dictionary(cls, dictionary: dict, gender):
        def return_if_exists(abreviation):
            if dictionary is not None:
                if abreviation in dictionary:
                    return dictionary[abreviation]
            return None
        obj = cls()
        obj.gender = gender
        obj.lved = return_if_exists('lved')
        obj.lves = return_if_exists('lves')
        obj.lvsv = return_if_exists('lvsv')
        obj.lved_i = return_if_exists('lved_i')
        obj.lves_i = return_if_exists('lves_i')
        obj.lvsv_i = return_if_exists('lvsv_i')

        obj.rved = return_if_exists('rved')
        obj.rves = return_if_exists('rves')
        obj.rvsv = return_if_exists('rvsv')
        obj.rved_i = return_if_exists('rved_i')
        obj.rves_i = return_if_exists('rves_i')
        obj.rvsv_i = return_if_exists('rvsv_i')
        return obj


# --------------------------------------
# Reorder percentages
# --------------------------------------

class Zone(Enum):
    UNK = 0   # unknown (missing data)
    AL = 1    # abnormal low
    NZ = 2    # normal zone
    AH = 3    # abnormal high


class ReorderPercentage:
    """
    Refrence: Petersen et al. Journal of Cardiovascular Magnetic Resonance (2017) 19:18 DOI 10.1186/s12968-017-0327-9
    """
    def __init__(self, volume_idcs: list):
        """
        volume_idcs - pair of VolumeIndices objects (original, predicted)
        """
        self.volume_idcs = volume_idcs

        self.zone_calculators = [
            self._lved, self._lves, self._lvsv,
            self._lved_idx, self._lves_idx, self._lvsv_idx,
            self._rved, self._rves, self._rvsv,
            self._rved_idx, self._rves_idx, self._rvsv_idx
        ]

    @staticmethod
    def _get_zone(gender, ventricular_value, male_ranges, female_ranges):
        if gender == Gender.M:
            if ventricular_value is None:
                return Zone.UNK
            for barrier, zone in zip(male_ranges, [Zone.AL, Zone.NZ]):
                if ventricular_value < barrier:
                    return zone
            return Zone.AH
        elif gender == Gender.F:
            if ventricular_value is None:
                return Zone.UNK
            for barrier, zone in zip(female_ranges, [Zone.AL, Zone.NZ]):
                if ventricular_value < barrier:
                    return zone
            return Zone.AH
        else:
            return Zone.UNK

    # Left side
    def _lved(self, volume_idcs: VolumeIndices):
        ventricular_value = volume_idcs.lved
        male_ranges = [93, 232]
        female_ranges = [80, 175]
        return self._get_zone(volume_idcs.gender, ventricular_value, male_ranges, female_ranges)

    def _lves(self, volume_idcs: VolumeIndices):
        ventricular_value = volume_idcs.lves
        male_ranges = [34, 103]
        female_ranges = [25, 73]
        return self._get_zone(volume_idcs.gender, ventricular_value, male_ranges, female_ranges)

    def _lvsv(self, volume_idcs: VolumeIndices):
        ventricular_value = volume_idcs.lvsv
        male_ranges = [49, 140]
        female_ranges = [47, 110]
        return self._get_zone(volume_idcs.gender, ventricular_value, male_ranges, female_ranges)

    def _lved_idx(self, volume_idcs: VolumeIndices):
        ventricular_value = volume_idcs.lved_i
        male_ranges = [52, 117]
        female_ranges = [50, 101]
        return self._get_zone(volume_idcs.gender, ventricular_value, male_ranges, female_ranges)

    def _lves_idx(self, volume_idcs: VolumeIndices):
        ventricular_value = volume_idcs.lves_i
        male_ranges = [19, 52]
        female_ranges = [16, 43]
        return self._get_zone(volume_idcs.gender, ventricular_value, male_ranges, female_ranges)

    def _lvsv_idx(self, volume_idcs: VolumeIndices):
        ventricular_value = volume_idcs.lvsv_i
        male_ranges = [28, 70]
        female_ranges = [29, 63]
        return self._get_zone(volume_idcs.gender, ventricular_value, male_ranges, female_ranges)

    # Right side
    def _rved(self, volume_idcs: VolumeIndices):
        ventricular_value = volume_idcs.rved
        male_ranges = [99, 260]
        female_ranges = [83, 192]
        return self._get_zone(volume_idcs.gender, ventricular_value, male_ranges, female_ranges)

    def _rves(self, volume_idcs: VolumeIndices):
        ventricular_value = volume_idcs.rves
        male_ranges = [34, 135]
        female_ranges = [26, 95]
        return self._get_zone(volume_idcs.gender, ventricular_value, male_ranges, female_ranges)

    def _rvsv(self, volume_idcs: VolumeIndices):
        ventricular_value = volume_idcs.rvsv
        male_ranges = [54, 140]
        female_ranges = [47, 107]
        return self._get_zone(volume_idcs.gender, ventricular_value, male_ranges, female_ranges)

    def _rved_idx(self, volume_idcs: VolumeIndices):
        ventricular_value = volume_idcs.rved_i
        male_ranges = [55, 128]
        female_ranges = [51, 110]
        return self._get_zone(volume_idcs.gender, ventricular_value, male_ranges, female_ranges)

    def _rves_idx(self, volume_idcs: VolumeIndices):
        ventricular_value = volume_idcs.rves_i
        male_ranges = [19, 67]
        female_ranges = [16, 55]
        return self._get_zone(volume_idcs.gender, ventricular_value, male_ranges, female_ranges)

    def _rvsv_idx(self, volume_idcs: VolumeIndices):
        ventricular_value = volume_idcs.rvsv_i
        male_ranges = [30, 69]
        female_ranges = [29, 61]
        return self._get_zone(volume_idcs.gender, ventricular_value, male_ranges, female_ranges)

    def reordering_percentage(self):
        """
        This function calculates how many times 
        the suggested zone is different
        in case of the predicted volume data.
        """
        overall_errors = {}
        LN = {}
        NH = {}
        NL = {}
        HN = {}
        LH = {}
        HL = {}
        for zone_calculator in self.zone_calculators:
            zc = lambda vi: (zone_calculator(vi[0]), zone_calculator(vi[1]))  # original, predicted
            volumes_as_zone = list(map(zc, self.volume_idcs))

            cntr = 0
            equal, ln, nh, nl, hn, lh, hl = 0, 0, 0, 0, 0, 0, 0
            for volume_pair in volumes_as_zone:
                if not(volume_pair[0] == Zone.UNK or volume_pair[1] == Zone.UNK):
                    cntr += 1
                    if volume_pair[0] == volume_pair[1]:
                        equal += 1
                    elif volume_pair[0] == Zone.AL and volume_pair[1] == Zone.NZ:
                        ln += 1
                    elif volume_pair[0] == Zone.NZ and volume_pair[1] == Zone.AH:
                        nh += 1
                    elif volume_pair[0] == Zone.NZ and volume_pair[1] == Zone.AL:
                        nl += 1
                    elif volume_pair[0] == Zone.AH and volume_pair[1] == Zone.NZ:
                        hn += 1
                    elif volume_pair[0] == Zone.AL and volume_pair[1] == Zone.AH:
                        lh += 1
                    elif volume_pair[0] == Zone.AH and volume_pair[1] == Zone.AL:
                        hl += 1
            
            overall_errors[zone_calculator.__name__] = (1 - equal / cntr) if cntr > 0 else None
            LN[zone_calculator.__name__] = (ln / cntr) if cntr > 0 else None
            NH[zone_calculator.__name__] = (nh / cntr) if cntr > 0 else None
            NL[zone_calculator.__name__] = (nl / cntr) if cntr > 0 else None
            HN[zone_calculator.__name__] = (hn / cntr) if cntr > 0 else None
            LH[zone_calculator.__name__] = (lh / cntr) if cntr > 0 else None
            HL[zone_calculator.__name__] = (hl / cntr) if cntr > 0 else None
        return overall_errors, LN, NH, NL, HN, LH, HL
