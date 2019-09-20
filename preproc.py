from skimage.transform import resize, rotate
from skimage.util import crop
from math import floor, ceil
import numpy as np


class Box:

    def __init__(self, xl, xr, yu, yb):
        self.xl = xl
        self.xr = xr
        self.yu = yu
        self.yb = yb
    
    def __getitem__(self, index):
        if index == 0:
            return self.xl
        elif index == 1:
            return self.xr
        elif index == 2:
            return self.yu
        elif index == 3:
            return self.yb
        else:
            return None
    
    @classmethod
    def from_relative(cls, image_size, center, box_size):
        ih, iw = image_size
        ch, cw = center
        h, w = box_size

        xl = int(iw * (cw - w / 2.0))
        xr = int(iw * w) + xl
        yu = int(ih * (ch - h / 2.0))
        yb = int(ih * h) + yu
        return cls(xl, xr, yu, yb)
    
    @classmethod
    def from_absolute(cls, center, box_size):
        ch, cw = center
        h, w = box_size

        xl = int(cw - w / 2.0)
        xr = int(w + xl)
        yu = int(ch - h / 2.0)
        yb = int(h + yu)
        return cls(xl, xr, yu, yb)
    
    @classmethod
    def create_default(cls, image_size):
        h, w = image_size
        return cls(0, w, 0, h)
    
    def shift_box(self, shift):
        dx, dy = shift
        self.xl += dx
        self.xr += dx
        self.yu += dy
        self.yb += dy
    
    def is_in(self, other):
        inside = True
        inside = inside and (self.xl >= other.xl)
        inside = inside and (self.xr < other.xr)
        inside = inside and (self.yu >= other.yu)
        inside = inside and (self.yb < other.yb)
        return inside
    
    def as_tuple(self):
        return int(self.xl), int(self.xr), int(self.yu), int(self.yb)


class Proc:

    def preprocess(self, image, contours):
        return image, contours, 1.0
    
    @staticmethod
    def cast_type(image, contours, dtype_image=None, dtype_contour=None):
        if dtype_image is not None:
            image = image.astype(dtype_image)
        if dtype_contour is not None:
            contours = [c.astype(dtype_contour) for c in contours]
        return image, contours
    
    @staticmethod
    def duplicate(image, contours):
        """
        All of the operators change the original input permanently.
        image - 2D np matrix, dtype: any
        contours - list of 2D matrices (N, 2), dtype: any
        """
        image2 = image.copy()
        contours2 = [c.copy() for c in contours]
        return image2, contours2
    
    @staticmethod
    def convert2utf8(image):
        image = Proc.intensity_rescale(image)
        image = image * 255
        return image.astype(np.uint8)
    
    @staticmethod
    def roi_box(image, contours, border):
        """
        Finds the ROI over the contours and a border is added to it
        in order to take into account the context.
        The function cares whether the ROI fits inside image and alters it accordingly.
        """
        def bbox(contour):
            # contour - matrix (N, 2)
            x0 = np.min(contour[:, 0])  # top left corner of the bounding box
            y0 = np.min(contour[:, 1])  # top left corner
            x1 = np.max(contour[:, 0])  # right bottom corner
            y1 = np.max(contour[:, 1])  # right bottom corner
            return (x0, y0, x1, y1)
        x0, y0, x1, y1 = None, 0, 0, 0  # the overall bbox 
        for contour in contours:
            box = bbox(contour)
            if x0 is None:
                x0, y0, x1, y1 = box
            else:
                x0 = min(x0, box[0])
                y0 = min(y0, box[1])
                x1 = max(x1, box[2])
                y1 = max(y1, box[3])
        xl = max(x0 - border, 0)
        yu = max(y0 - border, 0)
        xr = min(x1 + border, image.shape[1]-1)
        yb = min(y1 + border, image.shape[0]-1)
        return Box(xl, xr, yu, yb)
    
    @staticmethod
    def roi_box_with_fixed_size(image, contours, size):
        """
        Finds the ROI over the contours and a border is added to it
        in order to take into account the context.
        The function cares whether the ROI fits inside image and alters it accordingly.
        """
        def bbox(contour):
            # contour - matrix (N, 2)
            x0 = np.min(contour[:, 0])  # top left corner of the bounding box
            y0 = np.min(contour[:, 1])  # top left corner
            x1 = np.max(contour[:, 0])  # right bottom corner
            y1 = np.max(contour[:, 1])  # right bottom corner
            return (x0, y0, x1, y1)
        x0, y0, x1, y1 = None, 0, 0, 0  # the overall bbox 
        for contour in contours:
            box = bbox(contour)
            if x0 is None:
                x0, y0, x1, y1 = box
            else:
                x0 = min(x0, box[0])
                y0 = min(y0, box[1])
                x1 = max(x1, box[2])
                y1 = max(y1, box[3])

        cw = (x0 + x1) // 2
        ch = (y0 + y1) // 2
        sh, sw = size

        xl = cw - sw // 2
        yu = ch - sh // 2
        xr = cw + sw // 2
        yb = ch + sh // 2

        xl = max(xl, 0)
        yu = max(yu, 0)
        xr = min(xr, image.shape[1]-1)
        yb = min(yb, image.shape[0]-1)
        return Box(xl, xr, yu, yb)
    
    @staticmethod
    def roi_box_center(image, contours, size, center):
        """
        Finds the ROI around a given center with a fixed size.
        The function cares whether the ROI fits inside the image and alters it accordingly.
        """
        ch, cw = center
        sh, sw = size

        xl = cw - sw // 2
        yu = ch - sh // 2
        xr = cw + sw // 2
        yb = ch + sh // 2

        xl = max(xl, 0)
        yu = max(yu, 0)
        xr = min(xr, image.shape[1]-1)
        yb = min(yb, image.shape[0]-1)
        return Box(xl, xr, yu, yb)
    
    @staticmethod
    def bounding_box(image, contours, box_size):
        """
        Finds the bounding box around the contours with a box_size.
        The box is aligned to the center of the shapes.
        The function does not care whether the bounding box fits inside the image and alters it accordingly.
        """
        def bbox(contour):
            # contour - matrix (N, 2)
            x0 = np.min(contour[:, 0])  # top left corner of the bounding box
            y0 = np.min(contour[:, 1])  # top left corner
            x1 = np.max(contour[:, 0])  # right bottom corner
            y1 = np.max(contour[:, 1])  # right bottom corner
            return (x0, y0, x1, y1)
        x0, y0, x1, y1 = None, 0, 0, 0  # the overall bbox 
        for contour in contours:
            box = bbox(contour)
            if x0 is None:
                x0, y0, x1, y1 = box
            else:
                x0 = min(x0, box[0])
                y0 = min(y0, box[1])
                x1 = max(x1, box[2])
                y1 = max(y1, box[3])

        cw = (x0 + x1) // 2
        ch = (y0 + y1) // 2
        sh, sw = box_size

        xl = cw - sw // 2
        yu = ch - sh // 2
        xr = cw + sw // 2
        yb = ch + sh // 2
        return Box(xl, xr, yu, yb)
    
    @staticmethod
    def absolute2relative_coordinates(image, contours):
        """
        Normalize the coordinates relative to the image size.
        Warning: Input contours should not be already normalized!
        image - 2D np matrix, dtype: any
        contours - list of 2D matrices (N, 2), dtype: float
        """
        h, w = image.shape
        for c in contours:
            c[:, 0] = c[:, 0] / w
            c[:, 1] = c[:, 1] / h
        return image, contours
    
    @staticmethod
    def intensity_rescale(image, thresholds=(1.0, 99.0)):
        """ 
        Rescale the image intensity to the range of [0, 1].
        image - 2D np matrix, dtype: float
        """
        val_l, val_h = np.percentile(image, thresholds)
        image[image < val_l] = val_l
        image[image > val_h] = val_h
        image = (image - val_l) / (val_h - val_l + 1e-5)
        return image
    
    @staticmethod
    def gauss_noise(image, sigma=0.01, crop_intensity_intervall=(0.0, 1.0)):
        """
        Adds Gauss noise to the image. The noise has 0 mean.
        image - 2D np matrix, dtype: float
        sigma - size of noise
        crop_intensity_intervall - the noise can cause higher value than maximum intensity (or lower ...)
        """
        image = image + sigma * np.random.randn(*image.shape)
        if crop_intensity_intervall is not None:
            low, high = crop_intensity_intervall
            image[image < low] = low
            image[image > high] = high
        return image
    
    @staticmethod
    def crop(image, contours, ratio, crop_box):
        """
        Crops the image and transforms the contours according to crop_box.
        image - 2D np matrix, dtype: any
        contrours - list of 2D matrices (N, 2), dtype: any
        crop_box - Box 
        """
        xl, xr, yu, yb = crop_box.as_tuple()
        image2 = image[yu: yb, xl: xr]
        # shift the contour coordinates according to the crop
        for c in contours:
            c[:, 0] = c[:, 0] - xl
            c[:, 1] = c[:, 1] - yu
        return image2, contours, ratio
    
    @staticmethod
    def scale(image, contours, ratio, size_to_scale):
        """
        Up or down samples an image and transforms the contours respectively.
        image - 2D np matrix, dtype: any
        contrours - list of 2D matrices (N, 2), dtype: float
        size_to_scale - tuple(final size height, final size width)
        """
        sizeh, sizew = size_to_scale
        h, w = image.shape
        image = resize(image, (sizeh, sizew), preserve_range=True)
        scaler = (sizeh / h, sizew / w)
        for c in contours:
            c[:, 0] = c[:, 0] * scaler[1]
            c[:, 1] = c[:, 1] * scaler[0]
        return image, contours, ratio / (scaler[0] * scaler[1])
    
    @staticmethod
    def rotate(image, contours, ratio, angle):
        """
        Rotates the image and the contours around the center of the image.
        image - 2D np matrix, dtype: any
        contrours - list of 2D matrices (N, 2), dtype: float
        angle - in degree, positive means rotation is anti-clockwise
        """
        # rotate the image
        r_image = rotate(image, angle, preserve_range=True)
        r_cons = []
        # rotate contours one-by-one
        center = np.array([image.shape[1] / 2 - 0.5, image.shape[0] / 2 - 0.5])
        for con in contours:
            con = con - center
            phi = angle / 360 * 2 * np.pi
            R = np.array([
                    [np.cos(phi), np.sin(phi)],
                    [-np.sin(phi), np.cos(phi)]
                ])
            rotated_T = np.matmul(R, con.T)
            rotated_con = (rotated_T.T + center)
            r_cons.append(rotated_con)
        return r_image, r_cons, ratio
    
    @staticmethod
    def pad(image, contours, ratio, padding, value=0):
        """
        Puts zeros at around image and shifts the contours
        according to the new size.
        image - 2D np matrix, dtype: any
        contrours - list of 2D matrices (N, 2), dtype: float
        ratio - the change in the area
        padding - tuple (upper pad, bottom pad, left pad, right pad)
        """
        h, w = image.shape
        pu, pb, pl, pr = padding
        padded_image = np.zeros((h + pu + pb, w + pl + pr), dtype=image.dtype)
        padded_image[pu:(pu+h), pl:(pl+w)] = image
        for c in contours:
            c[:, 0] = c[:, 0] + pl
            c[:, 1] = c[:, 1] + pu
        return padded_image, contours, ratio
    
    @staticmethod
    def cropwithpad(image, contours, ratio, crop_box):
        """
        Crops the image and transforms the contours according to crop_box.
        If the crop_box does not fit into the image, first the image is padded.
        image - 2D np matrix, dtype: any
        contrours - list of 2D matrices (N, 2), dtype: any
        crop_box - tuple(x_left, x_right, y_up, y_bottom)
        """
        img_box = Box.create_default(image.shape)
        if crop_box.is_in(img_box):
            image, contours, _ = Proc.crop(image, contours, 1.0, crop_box)
        else:
            pl = int(max(0, img_box[0] - floor(crop_box[0])))  # calculating the padding
            pr = int(max(0, ceil(crop_box[1]) - img_box[1]))
            pu = int(max(0, img_box[2] - floor(crop_box[2])))
            pb = int(max(0, ceil(crop_box[3]) - img_box[3]))
            image, contours, _ = Proc.pad(image, contours, 1.0, (pu, pb, pl, pr))
            crop_box.shift_box((pl, pu))  # the padded image might have a new origo
            image, contours, _ = Proc.crop(image, contours, 1.0, crop_box)
        return image, contours, ratio
    
    @staticmethod
    def horizontalflip(image, contours, ratio):
        """
        Flips the image and the corresponding contours horizontally.
        image - 2D np matrix, dtype: any
        contrours - list of 2D matrices (N, 2), dtype: any
        """
        _, w = image.shape
        image2 = image[:, ::-1]
        for c in contours:
            c[:, 0] = w - c[:, 0] - 1
        return image2, contours, ratio


class CropROIfromCenter(Proc):
    """
    Crops the ROI around a given center.
    The size of the image ROI is fixed.
    The edges of the images are taken into account.
    The final image is scaled up.
    """
    def __init__(self, out_size, roi_size, center):
        super(CropROIfromCenter, self).__init__()
        self.out_size = out_size
        self.roi_size = roi_size
        self.center = center
    
    def preprocess(self, image, contours):
        ratio = 1.0
        image, contours = Proc.duplicate(image, contours)
        image, contours = Proc.cast_type(image, contours, np.float32, np.float32)
        image = Proc.intensity_rescale(image)
        ch, cw = self.center
        sh, sw = self.roi_size
        xl = cw - sw // 2
        yu = ch - sh // 2
        xr = cw + sw // 2
        yb = ch + sh // 2
        crop_box = Box(xl, xr, yu, yb)
        image, contours, ratio = Proc.cropwithpad(image, contours, ratio, crop_box)
        image, contours, ratio = Proc.scale(image, contours, ratio, self.out_size)
        return image, contours, ratio


class CenterCropnoScale(Proc):
    """
    Crops around the center of an image. The sizes of the MR images
    are different therefore a small image is padded instead of 
    cropping.
    """
    def __init__(self, crop_size, relative=False):
        super(CenterCropnoScale, self).__init__()
        self.crop_size = crop_size
        self.relative = relative  # crop size is relative to the image size
    
    def preprocess(self, image, contours):
        ratio = 1.0
        image, contours = Proc.duplicate(image, contours)
        image, contours = Proc.cast_type(image, contours, np.float32, np.float32)
        image = Proc.intensity_rescale(image)
        if self.relative:
            crop_box = Box.from_relative(image.shape, (0.5, 0.5), self.crop_size)
        else:
            center = (image.shape[0] // 2, image.shape[1] // 2)
            crop_box = Box.from_absolute(center, self.crop_size)
        
        image, contours, ratio = Proc.cropwithpad(image, contours, ratio, crop_box)
        return image, contours, ratio


class AugmentwithPaddedCenterCrop(Proc):
    """
    Augmentation applies the following steps on the RAW input:
    - intensity rescaling
    - rotation
    - shifted crop (from center and uses padding if the size is not enough)
    - gauss noise (remains within the thresholds, max intesity will remain the same)
    - changes the contour coordinates to relative
    """
    def __init__(self, crop_size, angle, shift, sigma):
        super(AugmentwithPaddedCenterCrop, self).__init__()
        self.crop_size = crop_size
        self.angle = angle
        self.shift = shift
        self.sigma = sigma
    
    def preprocess(self, image, contours):
        angle_min, angle_max = self.angle
        shift_min, shift_max = self.shift
        image, contours = Proc.duplicate(image, contours)
        image, contours = Proc.cast_type(image, contours, np.float32, np.float32)
        image = Proc.intensity_rescale(image)
        # rotation
        angle = np.random.randint(angle_min, angle_max)
        image, contours, _ = Proc.rotate(image, contours, 1.0, angle)
        # shift and pad if necessary then crop
        shift = np.random.randint(shift_min, shift_max, size=2)
        center = (image.shape[0] // 2, image.shape[1] // 2)
        crop_box = Box.from_absolute(center, self.crop_size)
        crop_box.shift_box(shift)
        image, contours, _ = Proc.cropwithpad(image, contours, 1.0, crop_box)
        # add noise
        image = Proc.gauss_noise(image, self.sigma)
        image, contours = Proc.absolute2relative_coordinates(image, contours)
        return image, contours


class AugmentROI(Proc):
    """
    Augmentation applies the following steps:
    - rotation
    - shifted crop (from roi center)
    - gauss noise (remains within the thresholds, max intesity will remain the same)
    - changes the contour coordinates to relative
    Assumes the cropping can be done, the sizes are appropriate.
    """
    def __init__(self, roi_size, angle, shift, sigma):
        super(AugmentROI, self).__init__()
        self.roi_size = roi_size
        self.angle = angle
        self.shift = shift
        self.sigma = sigma
    
    def preprocess(self, image, contours):
        angle_min, angle_max = self.angle
        shift_min, shift_max = self.shift
        image, contours = Proc.duplicate(image, contours)
        image, contours = Proc.cast_type(image, contours, np.float32, np.float32)
        image = Proc.intensity_rescale(image)
        # rotation
        angle = np.random.randint(angle_min, angle_max)
        image, contours, _ = Proc.rotate(image, contours, 1.0, angle)
        # scale
        scalex, scaley = np.random.rand(2) * 0.1 + 1
        scaled_sizex = int(image.shape[0] * scalex)
        scaled_sizey = int(image.shape[1] * scaley)
        image, contours, _ = Proc.scale(image, contours, 1.0, (scaled_sizex, scaled_sizey))
        shift = np.random.randint(shift_min, shift_max, size=2)
        #crop_box = Proc.roi_box_with_fixed_size(image, contours, self.roi_size)
        crop_box = Proc.bounding_box(image, contours, self.roi_size)
        crop_box.shift_box(shift)
        image, contours, _ = Proc.cropwithpad(image, contours, 1.0, crop_box)
        image, contours, _ = Proc.scale(image, contours, 1.0, (224, 224))
        image = Proc.gauss_noise(image, self.sigma)
        image, contours = Proc.absolute2relative_coordinates(image, contours)
        return image, contours
