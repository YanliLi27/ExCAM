from main_func.main_components.evaluator_funcs2 import cam_evaluator_step
from main_func.main_components.utils.target_cam_calculation import target_cam_selection
from main_func.main_components.utils.default_args import default_args
from main_func.main_components.im_funcs import im_reader, maxmin_reader
import os


def evaluator_main2(cam_method, model, target_layer, target_dataset,
                    general_args=None, im_path:str='your_path', target_category=None):
    # step 0. preset args, if not using the complex functions
    if general_args is None:
        print('use defualt args')
        general_args = default_args()
    if general_args.target_category_flag==None:
        general_args.target_category_flag = target_category
        
    # step 1. im_read - or not
    if os.path.exists(im_path) and general_args.im_selection_mode != 'all':
        print('loading importance matrix with mode:{}'.format(general_args.im_selection_mode))
        im = im_reader(im_path, general_args.im_selection_mode)
        # im [num_classes, num_features]
        im = target_cam_selection(im, mode=general_args.im_selection_mode, extra=general_args.im_selection_extra)
        # im [num_classes, num_features]
    else:
        im = None
    
    # step 2. max-min normalization - or not
    if im_path and general_args.stat_maxmin_flag:
        data_max_value, data_min_value = maxmin_reader(im_path)
        data_max_value, data_min_value = data_max_value * 1.5, data_min_value * 1.5
    else:
        data_max_value, data_min_value = None, None
    

    # step 3. pred step
    cam_evaluator_step(cam_method, model, target_layer, target_dataset,  # required attributes
                        general_args, # .batch_size, .groups, .target_category_flag, .fold_order
                        # optional function:
                        im=im, data_max_value=data_max_value, data_min_value=data_min_value, remove_minus_flag=general_args.remove_minus_flag,
                        max_iter=general_args.max_iter)
