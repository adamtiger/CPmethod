from scipy.interpolate import splprep, splev
from utils import get_logger
from preproc import Proc
import numpy as np

logger = get_logger(__name__)


class RoiBox:

    @staticmethod
    def curves2roi(image, curves) -> np.ndarray:
        """
        :param curves: list of N x 2 matrices, (x, y)
        :return: control points, (CONTROL_NUM x 2)
        """
        # find the roi around the contours
        if len(curves) == 0:
            return None
        xl, xr, yu, yb = Proc.roi_box(image, curves, 0).as_tuple()
        roi = np.zeros((4), dtype=np.float32)
        roi[0] = (xl + xr) / 2.0  # cx
        roi[1] = (yu + yb) / 2.0  # cy
        roi[2] = yu - yb  # height
        roi[3] = xr - xl  # width
        return roi
    
    @staticmethod
    def roi2contour(roi):
        """
        roi - given in absolute coordinates
        """
        cx, cy, h, w = roi.astype(int)
        contour = np.zeros((h * 2 + w * 2, 2))
        # draw the contour
        xl = cx - w // 2
        xr = xl + w
        yu = cy - h // 2
        yb = yu + h
        cntr = 0
        for y in range(yu, yb):
            contour[cntr, 0] = xl
            contour[cntr, 1] = y
            cntr += 1
        for x in range(xl, xr):
            contour[cntr, 0] = x
            contour[cntr, 1] = yb - 1
            cntr += 1
        for y in range(yb, yu, -1):
            contour[cntr, 0] = xr - 1
            contour[cntr, 1] = y
            cntr += 1
        for x in range(xl, xr, -1):
            contour[cntr, 0] = x
            contour[cntr, 1] = yu
            cntr += 1
        return contour

    @staticmethod
    def roi2image(image, roi: np.ndarray) -> np.ndarray:
        """
        :param controls: CONTROL_NUM x 2, (x, y)
        :return: image: SIZE x SIZE matrix, (x, y)
        """
        xl = int(roi[0] - roi[3] / 2.0)
        xr = int(roi[0] + roi[3] / 2.0)
        yu = int(roi[1] - roi[2] / 2.0)
        yb = int(roi[1] + roi[2] / 2.0)
        # drawing the box to the image
        image[yu:(yb+1), xl] = 1
        image[yb, xl:(xr+1)] = 1
        image[yu:(yb+1), xr] = 1
        image[yu, xl:(xr+1)] = 1
        return image

    @staticmethod
    def roicenter2image(image, roi):
        cx = int(roi[0])
        cy = int(roi[1])
        # drawing the box to the image
        image[cy-3:cy+3, cx-3:cx+3] = 1
        return image


CONTROL_NUM = 40

class ControlPoints:

    @staticmethod
    def curve2controls_rot(curve: np.ndarray) -> np.ndarray:
        """
        :param curve: N x 2 matrix, (x, y)
        :return: control points, (CONTROL_NUM x 2)
        """
        # draw the curve to a plane with the same size as dcm
        controls = []
        step_size = curve.shape[0] / CONTROL_NUM
        for idx in range(CONTROL_NUM):
            control_idx = round(idx * step_size)
            if control_idx < curve.shape[0]:
                x = curve[control_idx, 0]
                y = curve[control_idx, 1]
                controls.append((x, y))
        return np.array(controls)

    @staticmethod
    def controls2curve(controls: np.ndarray) -> np.ndarray:
        """
        :param controls: CONTROL_NUM x 2, (x, y)
        :return: curve: N x 2 matrix, (x, y)
        """
        try:
            pts = controls
            tck, u = splprep(pts.T, u=None, s=0.0, per=1)
            u_new = np.linspace(u.min(), u.max(), 200)
            x_new, y_new = splev(u_new, tck, der=0)
            curve = np.minimum(np.array([x_new, y_new]).T, 1.0)
        except ValueError:  # sometimes spline fitting fails
            logger.error("Wrong curve. Can not fit curve.")
            curve = controls
        return curve
    
    @staticmethod
    def controls2image(image_size, controls: np.ndarray) -> np.ndarray:
        """
        :param controls: CONTROL_NUM x 2, (x, y)
        :return: image: SIZE x SIZE matrix, (x, y)
        """
        curve = ControlPoints.controls2curve(controls) * (image_size[0] - 1)
        curve = curve.astype(int)
        image = np.zeros(image_size)
        image[curve[:, 1], curve[:, 0]] = 1
        return image
