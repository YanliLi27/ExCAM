import numpy as np
import torch
from tqdm import tqdm
from cam_components.core.base_cam_analyzer import BaseCAM_A


class ScoreCAM_A(BaseCAM_A):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None,compute_input_gradient=False,
                 uses_gradients=True):
        super(
            ScoreCAM_A,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform,
            compute_input_gradient,
            uses_gradients)
        print('score cam is not recommanded at all as it is too slow, let alone compute twice')

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



