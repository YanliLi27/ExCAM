import cv2
import numpy as np
import torch
from cam_components.core.activations_and_gradients import ActivationsAndGradients
from cam_components.utils.svd_on_activations import get_2d_projection
from torch import nn
from scipy.special import softmax


class BaseCAM_A:
    def __init__(self,
                 model,
                 target_layers,
                 use_cuda=False,
                 reshape_transform=None,
                 compute_input_gradient=False,
                 uses_gradients=True
                 ):
        self.model = model.eval()
        self.target_layers = target_layers
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)
        self.softamx = nn.Softmax(dim=1)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    def get_cam_weights(self,
                        input_tensor,
                        target_layers,
                        target_category,
                        activations,
                        grads,
                        ):
        raise Exception("Not Implemented")

    def get_loss(self, output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads):
        weights = self.get_cam_weights(input_tensor, target_layer,
                                       target_category, activations, grads)
        # print the max of weights and activations
        # print('max of the weights:', torch.max(weights))
        # print('max of the activations: ', torch.max(activations))
        if len(weights.shape)==2:
            weighted_activations = weights[:, :, None, None] * activations
            # weighted_activations -- [batch, channel, width, length]
        elif len(weights.shape)==4:
            weighted_activations = weights * activations
        else:
            raise ValueError('the length of weights is not valid')
        grad_cam = weighted_activations.sum(axis=1)  # from [batch, channel, length, width] to [batch, length, width]
        cam_grad_max_value = np.max(grad_cam, axis=(1, 2)).flatten()
        cam_grad_min_value = np.min(grad_cam, axis=(1, 2)).flatten()
        
        channel_numbers = weighted_activations.shape[1]   # weighted_activations[0] = [channel, length, width] numpy array
        B = weighted_activations.shape[0]
        cam = weighted_activations.sum(axis=(2, 3))  

        return cam, cam_grad_max_value, cam_grad_min_value  # cam [batch, all_channels]

    def forward(self, input_tensor, target_category=None):
        if self.cuda:
            input_tensor = input_tensor.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)

        output = self.activations_and_grads(input_tensor)
        if isinstance(target_category, int):
            target_category = [target_category] * input_tensor.size(0)
        
        np_output = output.cpu().data.numpy()
        predict_category = np.argmax(np_output, axis=-1)
        prob_predict_category = softmax(np_output, axis=-1)  # [batch*[2/1000 classes_normalized]]
        if target_category is None:
            target_category = predict_category
            pred_scores = np.max(prob_predict_category, axis=-1)
        else:
            assert(len(target_category) == input_tensor.size(0))
            matrix_zero = np.zeros([1, prob_predict_category.shape[-1]], dtype=np.int8)
            matrix_zero[0][target_category] = 1
            prob_predict_category = matrix_zero * prob_predict_category
            pred_scores = np.max(prob_predict_category, axis=-1)

        if self.uses_gradients:
            self.model.zero_grad()
            loss = self.get_loss(output, target_category)
            loss.backward(retain_graph=True)

        cam_per_layer, cam_grad_max_value, cam_grad_min_value = self.compute_cam_per_layer(input_tensor,
                                                                                           target_category)
        # list[target_layers,(array[batch, all_channel=512])]
        return cam_per_layer, predict_category, pred_scores, cam_grad_max_value, cam_grad_min_value  # 由于batch=1，target_layers=1， [1, 1, all_channels]

    def get_target_width_height(self, input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(
            self,
            input_tensor,
            target_category):
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        # Loop over the saliency image from every layer
        cam_importance_matrix = []
        for target_layer, layer_activations, layer_grads in \
                zip(self.target_layers, activations_list, grads_list):
            cam, cam_grad_max_value, cam_grad_min_value = self.get_cam_image(input_tensor,
                                                                            target_layer,
                                                                            target_category,
                                                                            layer_activations,
                                                                            layer_grads) 
            # cam = [batch, all_channels]
            cam_importance_matrix.append(cam)  # list [target_layers, (array[batch, all_channels])]

        return cam_importance_matrix, cam_grad_max_value, cam_grad_min_value
        # list[target_layers,(array[batch, channel, length, width])]
        # to -  1(target_layers/batch) * all channels(256 for each - 2* 256))

    def aggregate_multi_layers(self, cam_per_target_layer):  # 当target layer不止一层时采用平均的方式合成到一起
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    def scale_cam_image(self, cam, target_size=None):
        result = []
        for img in cam: 
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))  # 此处是因为np.min - np.min已经为0了
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

    def __call__(self,
                 input_tensor,
                 target_category=None):

        return self.forward(input_tensor,
                            target_category)  # [cam, predict_category]

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True
