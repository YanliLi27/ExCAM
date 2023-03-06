import numpy as np
from cam_components.core.base_cam_predictor import BaseCAM_P


class GradCAMPP_P(BaseCAM_P):
    def __init__(self, model, target_layers, importance_matrix, use_cuda=False, groups=1,
                 reshape_transform=None, compute_input_gradient=False, uses_gradients=True,
                 value_max=None, value_min=None, remove_minus_flag=False, out_logit=False):
        super(
            GradCAMPP_P,
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


