import cv2
import numpy as np
import torch
from cam_components.core.activations_and_gradients import ActivationsAndGradients
from cam_components.utils.svd_on_activations import get_2d_projection
from torch import nn
from scipy.special import softmax


class BaseCAM_P:
    def __init__(self,
                 model,
                 target_layers,
                 importance_matrix,
                 use_cuda:bool=False,
                 groups:int=2,
                 reshape_transform=None,
                 compute_input_gradient:bool=False,
                 uses_gradients:bool=True,
                 value_max=None,
                 value_min=None,
                 remove_minus_flag:bool=False,
                 out_logit:bool=False
                 ):
        if value_max:
            self.value_max = value_max
        else:
            self.value_max = None
        if value_min:
            self.value_min = value_min
        else:
            self.value_min = None
        self.model = model.eval()
        self.target_layers = target_layers
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.groups = groups
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.im = importance_matrix  # [num_classes, channels]
        self.uses_gradients = uses_gradients
        self.remove_minus_flag = remove_minus_flag
        self.out_logit = out_logit
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
        B, C, L, D = activations.shape

        if len(weights.shape)==2:
            weighted_activations = weights[:, :, None, None] * activations
            # weighted_activations -- [batch, channel, width, length]
        elif len(weights.shape)==4:
            weighted_activations = weights * activations
        else:
            raise ValueError('the length of weights is not valid')

        im_weights = np.zeros([B, C])
        if self.im is not None:  # if self.im not exist, use the original
            im_weights[:] = self.im[target_category]  # self.im [num_classes, all_channels] - im_weights [batch_size, all_channels]
        # im_weights [batch_size, channels] 
            weighted_activations = im_weights[:, :, None, None] * weighted_activations  # [batch, im-channel, None, None] * [batch, channel, length, width]
        channel_numbers = weighted_activations.shape[1]   # weighted_activations[0] = [channel, length, width] numpy array
        # print('channel_number:{}'.format(channel_numbers))
        channel_per_group = channel_numbers // self.groups
        # print('channel_per_group:{}'.format(channel_per_group))
        [B, C, L, D] = weighted_activations.shape
        # print('B,C,L,D:{}'.format(weighted_activations.shape))
        target_type = weighted_activations.dtype
        cam = np.zeros([B, self.groups, L, D], dtype=target_type) 
        for j in range (B):
            for i in range(self.groups):
                cam[j, i, :] = weighted_activations[j, i*channel_per_group:(i+1)*channel_per_group, :].sum(axis=0)
                # print('max of the group:{}'.format(cam[j, i, :].max()))
        return cam  # group:[batch, groups=2, length, width], while original: [batch, length, width]

    def forward(self, input_tensor, gt, target_category=None):
        if self.cuda:
            input_tensor = input_tensor.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)

        output = self.activations_and_grads(input_tensor)
        
        np_output = output.cpu().data.numpy()
        prob_predict_category = softmax(np_output, axis=-1)  # [batch*[2/1000 classes_normalized]]
        predict_category = np.argmax(prob_predict_category, axis=-1)
        if target_category is None:
            target_category = predict_category
            if self.out_logit:
                pred_scores = np.max(np_output, axis=-1)
                nega_scores = np.sum(np_output, axis=-1)
            else:
                pred_scores = np.max(prob_predict_category, axis=-1)
                nega_scores = None
        elif isinstance(target_category, int):
            target_category = [target_category] * input_tensor.size(0)
            assert(len(target_category) == input_tensor.size(0))
            
            if self.out_logit:
                matrix_zero = np.zeros([len(np_output), prob_predict_category.shape[-1]], dtype=np.int8)
                matrix_zero[:][target_category] = 1
                pred_scores = np.max(matrix_zero * np_output, axis=-1)
                nega_scores = np.sum(np_output, axis=-1)
            else:
                matrix_zero = np.zeros([len(np_output), prob_predict_category.shape[-1]], dtype=np.int8)  # TODO for parallel, change the 1/0 to batch size
                matrix_zero[:][target_category] = 1
                prob_predict_category = matrix_zero * prob_predict_category
                pred_scores = np.max(prob_predict_category, axis=-1)
                nega_scores = None
        
        elif target_category == 'GT':
            target_category = gt.to('cpu').data.numpy().astype(int)
            matrix_zero = np.zeros([len(np_output), prob_predict_category.shape[-1]], dtype=np.int8)
            matrix_zero[list(range(len(np_output))), target_category] = 1
            pred_scores = np.max(matrix_zero* np_output, axis=-1)
            nega_scores = np.sum(np_output, axis=-1)
        
        else:
            raise ValueError('not valid target_category')

        if self.uses_gradients:
            self.model.zero_grad()
            loss = self.get_loss(output, target_category)
            loss.backward(retain_graph=True)

        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   target_category,
                                                   prob_weights=prob_predict_category)

        # list[target_layers,(array[batch, channel, length, width])]
        # print('cam_per_later returned to the outer item: {}'.format(np.squeeze(cam_per_layer).shape))  # (array[batch, channel, length, width]) squeeze to remove the list -- and get the [batch, channel, length, width]
        # batch1[2, 512, 512], batch2.....

        return cam_per_layer, predict_category, pred_scores, nega_scores  # 由于batch=1，target_layers=1， [1, 1, groups, length, width]

    def get_target_width_height(self, input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(
            self,
            input_tensor,
            target_category,
            prob_weights=1.0):
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        cam_per_target_layer_per_batch = []
        # Loop over the saliency image from every layer

        for target_layer, layer_activations, layer_grads in \
                zip(self.target_layers, activations_list, grads_list):
            cam = self.get_cam_image(input_tensor,
                                     target_layer,
                                     target_category,
                                     layer_activations,
                                     layer_grads)
            # print('the cam from get_cam_image, should be [batch, groups=2, length, width], truth is: {}'.format(cam.shape))
            # cam = [batch, groups, length, width]
            # cam = np.squeeze(cam, axis=0)  # (1, 2, 64, 64) only works when the batch=1
            if self.remove_minus_flag:
                cam = np.maximum(cam, 0)
            for cam_item in cam:
                scaled = self.scale_cam_image(cam_item, target_size, prob_weights)  # 放缩不需要改变，不只是放缩比例，更重要的是进行了归一化
                # scaled [batch, groups, length, width]
                cam_per_target_layer_per_batch.append(scaled)
            cam_per_target_layer.extend(cam_per_target_layer_per_batch)  # list[target_layers,(array[batch, groups, length, width])]
                # cam_per_target_layer.append(scaled[:, None, :]) # 原本的scaled[:, None, :]相当于扩展出channel维

        return cam_per_target_layer # list[target_layers,(array[batch, channel, length, width])]
        # -  1(target_layers/batch) * array(2(groups), 512, 512)

    def scale_cam_image(self, cam, target_size=None, prob_weights=1.0):
        result = []
        # cam [groups, length, width]       
        # it's ok to have normalization inside each groups. -- but we need the prob_weights to fix it.
        for img in cam:  # [length, width] don't calculate the max and min of each groups, we need the real ratio
            if self.value_max and self.value_min:
                value_max = self.value_max
                if self.remove_minus_flag:
                    if value_max > 0:
                        value_min = 0
                    else:
                        value_min = self.value_min
                else:
                    value_min = self.value_min
            else:
                value_max = np.max(img) + 1e-7
                value_min = np.min(img)
            img = (img - value_min) / (value_max - value_min)
            # TODO 置信度修正
            # img = img * prob_weights  # 根据置信度进行修正
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

    def __call__(self,
                 input_tensor,
                 gt,
                 target_category=None):

        return self.forward(input_tensor,
                            gt,
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
