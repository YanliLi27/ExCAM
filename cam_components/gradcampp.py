import numpy as np
from cam_components.core.base_cam_analyzer import BaseCAM_A


class GradCAMPP_A(BaseCAM_A):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None,compute_input_gradient=False,
                 uses_gradients=True):
        super(
            GradCAMPP_A,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform,
            compute_input_gradient,
            uses_gradients)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads,
                        ):
        grads_power_2 = grads**2
        grads_power_3 = grads_power_2 * grads
        # Equation 19 in https://arxiv.org/abs/1710.11063
        sum_activations = np.sum(activations, axis=(2, 3))
        eps = 0.000001
        aij = grads_power_2 / (2 * grads_power_2 +
                               sum_activations[:, :, None, None] * grads_power_3 + eps)
        # Now bring back the ReLU from eq.7 in the paper,
        # And zero out aijs where the activations are 0
        aij = np.where(grads != 0, aij, 0)
        weights = np.maximum(grads, 0) * aij
        return np.sum(weights, axis=(2, 3))



