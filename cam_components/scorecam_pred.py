import numpy as np
import torch
from tqdm import tqdm
from cam_components.core.base_cam_predictor import BaseCAM_P


class ScoreCAM_P(BaseCAM_P):
    def __init__(self, model, target_layers, importance_matrix, use_cuda=False, groups=1,
                 reshape_transform=None, compute_input_gradient=False, uses_gradients=True,
                 value_max=None, value_min=None, remove_minus_flag=False, out_logit=False):
        super(
            ScoreCAM_P,
            self).__init__(
            model,
            target_layers,
            importance_matrix,
            use_cuda,
            groups,
            reshape_transform,
            compute_input_gradient,
            uses_gradients,
            value_max=value_max,
            value_min=value_min,
            remove_minus_flag=remove_minus_flag,
            out_logit=out_logit)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads,
                        ):
        with torch.no_grad():
            upsample = torch.nn.UpsamplingBilinear2d(
                size=input_tensor.shape[-2:])
            activation_tensor = torch.from_numpy(activations)
            if self.cuda:
                activation_tensor = activation_tensor.cuda()

            upsampled = upsample(activation_tensor)

            maxs = upsampled.view(upsampled.size(0),
                                  upsampled.size(1), -1).max(dim=-1)[0]
            mins = upsampled.view(upsampled.size(0),
                                  upsampled.size(1), -1).min(dim=-1)[0]
            maxs, mins = maxs[:, :, None, None], mins[:, :, None, None]
            upsampled = (upsampled - mins) / (maxs - mins)

            input_tensors = input_tensor[:, None,
                                         :, :] * upsampled[:, :, None, :, :]

            if hasattr(self, "batch_size"):
                BATCH_SIZE = self.batch_size
            else:
                BATCH_SIZE = 16

            scores = []
            for batch_index, tensor in enumerate(input_tensors):
                category = target_category[batch_index]
                for i in tqdm.tqdm(range(0, tensor.size(0), BATCH_SIZE)):
                    batch = tensor[i: i + BATCH_SIZE, :]
                    outputs = self.model(batch).cpu().numpy()[:, category]
                    scores.extend(outputs)
            scores = torch.Tensor(scores)
            scores = scores.view(activations.shape[0], activations.shape[1])

            weights = torch.nn.Softmax(dim=-1)(scores).numpy()
            return weights


