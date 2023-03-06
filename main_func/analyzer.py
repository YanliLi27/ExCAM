from main_func.main_components.analyzer_funcs import cam_stats_step
from main_func.main_components.im_funcs import im_save


def analyzer_main(cam_method, model, target_layer, target_dataset, im_path, num_out_channel, num_classes, general_args):
    im_overall, im_target, im_diff, cam_grad_max_matrix, cam_grad_min_matrix \
        = cam_stats_step(cam_method, target_layer, general_args, # for the cam setting
                        model, target_dataset, num_out_channel, num_classes, # for the model and dataset
                        )
    
    im_save(im_overall, im_target, im_diff, 
            cam_grad_max_matrix, cam_grad_min_matrix,
            im_path)

