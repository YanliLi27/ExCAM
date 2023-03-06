import os


def weights_path_creator(model_flag, task):
    '''
    can be replaced with any reasonable path creator
    '''
    if task=='CatsDogs':
        model_dir = f"D:\\CatsDogs\\kaggle\\working"
    elif task=='MNIST':
        model_dir = f"D:\\CatsDogs\\MNIST"
    elif task=='Imagenet':
        model_dir = f"D:\\CatsDogs\\Imagenet"
    elif task=='ClickMe':
        model_dir = f"D:\\CatsDogs\\ClickMe"
    else:
        raise AttributeError('task name not valid')
    resume_path = os.path.join(model_dir, f"{task}_{model_flag}_best_model.model")
    return resume_path


def im_path_creator(model_flag, task, dataset_split, general_args):
    im_dir = os.path.join('./output/im/', '{}_{}_{}'.format(task, dataset_split, model_flag))
    if not os.path.exists(im_dir):
        os.makedirs(im_dir)
    im_name = f'All_fold{general_args.fold_order}_im_cate{general_args.target_category_flag}_{general_args.method}.csv'   
    return os.path.join(im_dir, im_name)
    # ./output/im/MNIST_val_resnet/All_fold0_im_cateNone_gradmean.csv


def cam_dir_creator(model_flag, task, dataset_split, general_args):
    cam_father_dir = os.path.join('./output/cam/', '{}_{}_{}'.format(task, dataset_split, model_flag))
    cam_son_dir = f'fold{general_args.fold_order}_cate{general_args.target_category_flag}_mt{general_args.method}'
    cam_sub_dir = f'norm{general_args.stat_maxmin_flag}_rm0{general_args.remove_minus_flag}_{general_args.im_selection_mode}{general_args.im_selection_extra}'
    cam_dir = os.path.join(cam_father_dir, cam_son_dir, cam_sub_dir)
    if not os.path.exists(cam_dir):
        os.makedirs(cam_dir)
    return cam_dir

# {stat_maxmin_name}_{remove_minus_name}_{im_selection_mode}{im_selection_extra}


def path_generator(model_flag:str='resnet',
                   task:str='CatsDogs', dataset_split:str='val',
                   general_args=None):
    weights_path = weights_path_creator(model_flag, task)  # absolute path of the weights file

    im_path = im_path_creator(model_flag, task, dataset_split, general_args)

    cam_dir = cam_dir_creator(model_flag, task, dataset_split, general_args)

    return weights_path, im_path, cam_dir
