import numpy as np
from cam_components.core.base_cam_analyzer import BaseCAM_A


class XGradCAM_A(BaseCAM_A):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None,compute_input_gradient=False,
                 uses_gradients=True):
        super(
            XGradCAM_A,
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
        sum_activations = np.sum(activations, axis=(2, 3))
        eps = 1e-7
        weights = grads * activations / \
            (sum_activations[:, :, None, None] + eps)
        weights = weights.sum(axis=(2, 3))
        return weights


