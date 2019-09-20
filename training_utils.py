from datetime import datetime
from utils import get_logger
import numpy as np
import torch

logger = get_logger(__name__)


class One2OneOptimizer:

    def __init__(self, model, optim, criterion, loaders, epochs, results, scheduler=None):
        self.model = model
        self.optim = optim
        self.scheduler = scheduler
        self.criterion = criterion
        self.sample_num = len(loaders[0])
        self.loaders = loaders
        self.epochs = epochs
        self.results = results
        self.validation_num = 200
        self.weight_saving_num = 10

    def fit(self):
        logger.info("Started training at {}".format(datetime.now()))
        eval_freq = int(max((self.sample_num * self.epochs) // 100, 1))
        save_freq = int(max((self.sample_num * self.epochs) // 10, 1))
        global_counter = 0
        train_loader = self.loaders[0]
        self.results.visualize_graph(self.model, next(iter(train_loader))['image'])

        for epoch in range(self.epochs):
            for index, sample in enumerate(train_loader):
                global_counter += 1

                image = sample['image']
                target = sample['target']

                predicted = self.model(image)
                loss = self.criterion(predicted, target)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                train_loss = loss.cpu().detach().numpy()
                self.results.cache_train_loss(train_loss)

                if global_counter % eval_freq == 0:
                    self._validation(epoch, index, global_counter)

                if global_counter % save_freq == 0:
                    self.results.cache_model_weights(self.model)
                    self.results.create_checkpoint(global_counter, self.model, self.optim)
                print("\r Current batch {}".format(index), end="")
        print("Training has finished.")

    def _validation(self, epoch, index, global_counter):
        counter = 0
        loss_sum = 0.0
        loss2_sum = 0.0  # square of the loss
        val_loader = self.loaders[1]
        for sample in val_loader:
            counter += 1
            image = sample['image']
            target = sample['target']
            predicted = self.model(image)
            loss = self.criterion(predicted, target)

            temp = loss.cpu().detach().numpy()
            loss_sum += temp
            loss2_sum += temp * temp
            del loss
            target_cpu = target.cpu().detach().numpy()
            predicted_cpu = predicted.cpu().detach().numpy()
            if counter < 8:
                image_cpu = image.cpu().detach().numpy()
                self.results.cache_image(image_cpu[0, 0], target_cpu[0], predicted_cpu[0])
            self.results.cache_val_prediction(target_cpu, predicted_cpu)

        loss_mean = loss_sum / counter
        loss_std = np.sqrt(loss2_sum / counter - loss_mean ** 2)

        self.results.cache_val_loss(loss_mean)
        self.results.send2visualizer(global_counter)
        print("Current status at: (epoch: %d, i: %d ) with validation loss: %f +/- %f" %
              (epoch, index, loss_mean, np.sqrt(loss_std)))


class ConGradient(torch.autograd.Function):
    """
    This module calculates the gradient vectors at the 
    output directly. The real control points are compared
    to the new predicted control points. 
    """
    def __init__(self):
        super(ConGradient, self).__init__()
        self.loss = None
        self.pseudo_grad = None
    
    def forward(self, predicted, target):
        self.save_for_backward(predicted)
        rotated = ConGradient._rotate_contours(target)
        best_ordered = ConGradient._chose_best(predicted, rotated)
        self.pseudo_grad = self._point2triangle(best_ordered, predicted)
        return self.loss
    
    def backward(self, grad_output):
        grad = self.pseudo_grad
        return grad, torch.tensor(0)

    @staticmethod
    def _rotate_contours(contour):
        """
        This function creates a matrix with the contours
        rotated around.
        :param contour: contour represented by control points (N, 2)
        :return: rotated contours (N, N, 2)
        """
        N = contour.size(1)
        index_mtx = torch.tensor([idx for idx in range(N)], dtype=torch.long)
        index_mtx = index_mtx.unsqueeze(0).repeat(N, 1)
        index_mtx = index_mtx + index_mtx.transpose(0, 1)
        index_mtx = index_mtx.fmod(N)

        return contour[:, index_mtx, :]

    @staticmethod
    def _chose_best(predicted, rotated):
        """
        This function finds the best order of the predicted points.
        The best order means the closest points in terms of MSE.
        :param target: target the model should predict
        :param rotated: predicted control in different orders
        :return: predicted control points in an order which best fits to the target
        """
        N = predicted.size(1)
        temp_ = predicted.unsqueeze(1).repeat(1, N, 1, 1)
        temp_ = temp_ - rotated
        temp_ = temp_**2
        temp_ = torch.sum(temp_, (2, 3))
        best_idx = torch.argmin(temp_, dim=1)
        indices = torch.zeros(2, best_idx.size(0)).type(torch.long)
        indices[0, :] = torch.tensor([idx for idx in range(best_idx.size(0))], dtype=torch.long)
        indices[1, :] = best_idx
        return rotated[indices[0], indices[1], :, :]

    def _point2triangle(self, target, predicted):
        """
        This function calculates the gradient as the differences between the
        target and the prediction.
        :return gradient
        """
        N = predicted.size(1)
        # first shift the vector to left and right
        index_mtx = torch.tensor([idx for idx in range(N)], dtype=torch.long)
        index_mtx = index_mtx.unsqueeze(0).repeat(3, 1)  # left, 0, right
        index_mtx = index_mtx + torch.tensor([[-1], [0], [1]], dtype=torch.long)  # shift to left, nowhere, right
        index_mtx = index_mtx.remainder(N)
        # calculate the traingular's points as vectors
        triangle = target[:, index_mtx, :]
        v1 = triangle[:, 0, :, :] - triangle[:, 1, :, :]
        v2 = triangle[:, 2, :, :] - triangle[:, 1, :, :]
        r = predicted - target
        r_v1 = r - v1
        r_v2 = r - v2

        # calculate the negatives of the gradients
        # 1: calculates the distance from the triangle
        v1x, v1y = v1[:, :, 0], v1[:, :, 1]
        v2x, v2y = v2[:, :, 0], v2[:, :, 1]
        x, y = r[:, :, 0], r[:, :, 1]

        v1_sq = torch.sum(v1 * v1, dim=2)
        v2_sq = torch.sum(v2 * v2, dim=2)

        distance1 = torch.zeros_like(target)
        distance2 = torch.zeros_like(target)
        
        temp = (v1x * y - v1y * x)
        distance1[:, :, 0] = -v1y * temp / (v1_sq + 1e-5)
        distance1[:, :, 1] = v1x * temp / (v1_sq + 1e-5)

        temp = (v2x * y - v2y * x)
        distance2[:, :, 0] = -v2y * temp / (v2_sq + 1e-5)
        distance2[:, :, 1] = v2x * temp / (v2_sq + 1e-5) 

        d1 = torch.sum(distance1 * distance1, dim=2)
        d2 = torch.sum(distance2 * distance2, dim=2)

        r_v1_sq = torch.sum(r_v1 * r_v1, dim=2)
        r_v2_sq = torch.sum(r_v2 * r_v2, dim=2)
        r_distance = torch.sum(r * r, dim=2)
        
        # checking whether the point is inside or not
        d1_valid = (((r_distance - d1) < v1_sq) * ((r_v1_sq - d1) < v1_sq)).type(torch.cuda.FloatTensor).unsqueeze(2)
        d2_valid = (((r_distance - d2) < v2_sq) * ((r_v2_sq - d2) < v2_sq)).type(torch.cuda.FloatTensor).unsqueeze(2)
        
        both_valid = d1_valid * d2_valid
        none_valid = (1 - d1_valid) * (1 - d2_valid)
        gradient = distance1 * d1_valid + distance2 * d2_valid - 0.5 * (distance1 + distance2) * both_valid + r * none_valid
        self.loss = torch.sum(torch.sqrt(torch.sum(gradient * gradient, dim=2))) \
                    / (gradient.size(0) * gradient.size(1))

        return gradient \
               / torch.max(torch.sqrt(torch.sum(gradient * gradient, dim=2)).unsqueeze(2), torch.cuda.FloatTensor([1e-5]))
