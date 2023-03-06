import os
from generators.main_generator import main_generator
from main_func.analyzer import analyzer_main
from main_func.predictor import predictor_main
from main_func.evaluator import evaluator_main
from main_func.evaluator2 import evaluator_main2
from cam_components import GradCAM_A, GradCAM_P, FullCAM_A, FullCAM_P, GradCAMPP_A, GradCAMPP_P, XGradCAM_A, XGradCAM_P
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # general args
    parser.add_argument('--batch_size', type=int, default=16, help='Dataloader batch size')
    parser.add_argument('--groups', type=int, default=1, help='If use group convolution')
    parser.add_argument('--task', type=str, default='CatsDogs', choices=['CatsDogs', 'MNIST', 'Imagenet'],
                         help='tasks')  # for model, dataset, cam, and file management
    
    # for model
    parser.add_argument('--model_flag', type=str, default='resnet', choices=['resnet', 'vgg', 'scratch', 'scratch_mn'],
                         help='model type')
    parser.add_argument('--randomization', type=bool, default=False, help='If start randomization')
    parser.add_argument('--random_severity', type=int, default=0, choices=[0, 1, 2, 3, 4], help='n/4 randomization')

    # for file management and loop
    parser.add_argument('--fold_order', type=int, default=0, help='For cross validation')
    parser.add_argument('--dataset_split', type=str, default='val', choices=['train', 'val', 'test'],
                         help='split')


    # cam required args
    parser.add_argument('--target_category_flag', default=None, help='Output category')
    parser.add_argument('--method', type=str, default='gradcam', choices=['gradcam', 'fullcam', 'gradcampp', 'smoothgradcam'],
                        help='If use the mean of gradients')
    parser.add_argument('--confidence_weight_flag', type=bool, default=False, help='Output category')

    # optional function -- core improvements
    parser.add_argument('--use_stat', type=bool, default=True, help='For using analyzer and im')
    # for predictor
    parser.add_argument('--stat_maxmin_flag', type=bool, default=True, help='For maxmin normalization')
    parser.add_argument('--remove_minus_flag', type=bool, default=True, help='If remove the part of heatmaps less than 0')
    parser.add_argument('--im_selection_mode', type=str, default='diff_top', choices=['max', 'top', 'diff_top', 'freq', 'index', 'all'],
                        help='If use statistic')
    parser.add_argument('--im_selection_extra', type=float, default=0.05, help='attributes of selection mode')
    parser.add_argument('--max_iter', default=None, help='max iteration to save time')

    args = parser.parse_args()
    return args


def analyzer_process(method:str='gradcam', target_category=None, model_flag:str='resnet', task:str='CatsDogs', dataset_split:str='val'):
    general_args = get_args()
    # --------------------------------------- re-define for loop --------------------------------------- #
    # Select the CAM method and target_category (optional)
    general_args.method = method
    general_args.target_category_flag = target_category
    # The model, task(data) and dataset split
    general_args.model_flag = model_flag
    general_args.task = task
    general_args.dataset_split = dataset_split
    # --------------------------------------- re-define for loop --------------------------------------- #

    for i in range(0, 1):
        general_args.fold_order = i
        model, target_layer, target_dataset, im_path, cam_dir, out_channel, num_classes = \
                                                            main_generator(model_flag=general_args.model_flag,
                                                                            task=general_args.task,
                                                                            dataset_split=general_args.dataset_split,
                                                                            general_args=general_args,
                                                                            )
        # im_path: # ./output/im/MNIST_val_resnet/All_fold0_im_cateNone_Grad.csv

        cam_method_zoo = {"gradcam": [GradCAM_A, GradCAM_P], 
        "fullcam": [FullCAM_A, FullCAM_P],
        "gradcampp":[GradCAMPP_A, GradCAMPP_P],
        "xgradcam":[XGradCAM_A, XGradCAM_P]}
        cam_method = cam_method_zoo[general_args.method]
        analyzer_main(cam_method[0], model, target_layer, target_dataset, im_path, out_channel, num_classes, general_args) 
    

def predictor_process(method:str='gradcam', target_category=None, model_flag:str='resnet', task:str='CatsDogs', dataset_split:str='val',
                        randomization:bool=False, random_severity:int=0,
                        maxmin_flag:bool=True, remove_minus_flag:bool=True, im_selection_mode:str='diff_top', im_selection_extra:float=0.05,
                        max_iter=None, mm_ratio:float=1.0
                        ):
    general_args = get_args()
    # --------------------------------------- re-define for loop --------------------------------------- #
    # Select the CAM method and target_category (optional)
    general_args.method = method
    general_args.target_category_flag = target_category

    # Extra function of the CAM creator
    general_args.stat_maxmin_flag = maxmin_flag
    general_args.remove_minus_flag = remove_minus_flag
    general_args.im_selection_mode = im_selection_mode
    general_args.im_selection_extra = im_selection_extra
    general_args.max_iter = max_iter

    # The model, task(data) and dataset split
    general_args.model_flag = model_flag
    general_args.task = task
    general_args.dataset_split = dataset_split
    # 是否模型随机化
    general_args.randomization = randomization
    general_args.random_severity = random_severity

    # if use the original img: for mnist
    if task == 'MNIST':
        use_origin = False
    else:
        use_origin = True
    # --------------------------------------- re-define for loop --------------------------------------- #


    for i in range(0, 1):
        general_args.fold_order = i
        model, target_layer, target_dataset, im_path, cam_dir, in_channel, num_classes = \
                                                            main_generator(model_flag=general_args.model_flag,
                                                                            task=general_args.task,
                                                                            dataset_split=general_args.dataset_split,
                                                                            general_args=general_args,
                                                                            )
        # im_path: # ./output/im/MNIST_val_resnet/All_fold0_im_cateNone_Grad.csv

        cam_method_zoo = {"gradcam": [GradCAM_A, GradCAM_P], 
                        "fullcam": [FullCAM_A, FullCAM_P],
                        "gradcampp":[GradCAMPP_A, GradCAMPP_P],
                        "xgradcam":[XGradCAM_A, XGradCAM_P]}
        cam_method = cam_method_zoo[general_args.method]
        predictor_main(cam_method[1], model, target_layer, target_dataset, cam_dir, general_args, im_path, mm_ratio=mm_ratio, use_origin=use_origin)


def evaluator_process(method:str='gradcam', target_category=None, model_flag:str='resnet', task:str='CatsDogs', dataset_split:str='val',
                        randomization:bool=False, random_severity:int=0,
                        maxmin_flag:bool=True, remove_minus_flag:bool=True, im_selection_mode:str='diff_top', im_selection_extra:float=0.05,
                        max_iter=None, version:int=1,
                        ):
    general_args = get_args()
    # --------------------------------------- re-define for loop --------------------------------------- #
    # Select the CAM method and target_category (optional)
    general_args.method = method
    general_args.target_category_flag = target_category

    # Extra function of the CAM creator
    general_args.stat_maxmin_flag = maxmin_flag
    general_args.remove_minus_flag = remove_minus_flag
    general_args.im_selection_mode = im_selection_mode
    general_args.im_selection_extra = im_selection_extra
    if max_iter is not None:
        general_args.max_iter = max_iter

    # The model, task(data) and dataset split
    general_args.model_flag = model_flag
    general_args.task = task
    general_args.dataset_split = dataset_split
    # 是否模型随机化
    general_args.randomization = randomization
    general_args.random_severity = random_severity
    # --------------------------------------- re-define for loop --------------------------------------- #

    for i in range(0, 1):
        general_args.fold_order = i
        model, target_layer, target_dataset, im_path, cam_dir, in_channel, num_classes = \
                                                            main_generator(model_flag=general_args.model_flag,
                                                                            task=general_args.task,
                                                                            dataset_split=general_args.dataset_split,
                                                                            general_args=general_args,
                                                                            )
        # im_path: # ./output/im/MNIST_val_resnet/All_fold0_im_cateNone_Grad.csv
        print('im_path: {}'.format(im_path))
        cam_method_zoo = {"gradcam": [GradCAM_A, GradCAM_P], 
        "fullcam": [FullCAM_A, FullCAM_P],
        "gradcampp":[GradCAMPP_A, GradCAMPP_P],
        "xgradcam":[XGradCAM_A, XGradCAM_P]}
        cam_method = cam_method_zoo[general_args.method]
        if version == 1:
            evaluator_main(cam_method[1], model, target_layer, target_dataset, general_args, im_path)
        elif version == 2:
            evaluator_main2(cam_method[1], model, target_layer, target_dataset, general_args, im_path)


def naive_gradcam(target_category=0):
    cam_method = GradCAM_P
    from torchvision.models import resnet18
    model = resnet18(pretrained=True)
    target_layer = [model.layer4[-1]] 
    # or: from generators.models.utils.find_layer import locate_candidate_layer
    # target_layer = locate_candidate_layer(model, [3, 224, 224])
    from torchvision.io.image import read_image
    target_dataset = read_image("path/to/your/image.png")
    cam_dir = 'path/to/save/image.png'
    predictor_main(cam_method, model, target_layer, target_dataset, cam_dir, target_category=target_category)



if __name__ == '__main__':
    task = 'CatsDogs'  # 'CatsDogs', 'MNIST', 'Imagenet', 
    model_flag = 'resnet'  # 'vgg', 'resnet', 'scratch', 'scratch_mnist'
    methods = ['gradcam', 'fullcam', 'gradcampp', 'xgradcam']

    for method in methods:
        analyzer_process(method=method, target_category=1, model_flag=model_flag, task=task, dataset_split='val')

    maxmin = [True, False]
    rmove = [True, False]
    immode = ['all', 'diff_top']

    
    for method in methods:
        for mm in maxmin:
            for rm in rmove:
                for im in immode:
                    predictor_process(method=method, target_category=1, model_flag=model_flag, task=task, dataset_split='val',
                                        randomization=False, random_severity=0,
                                        maxmin_flag=mm, remove_minus_flag=rm, im_selection_mode=im, im_selection_extra=0.05,
                                        max_iter=None, mm_ratio=1
                                       )
                    # evaluator_process(method=method, target_category=None, model_flag='vgg', task='Imagenet', dataset_split='val',
                    #                     randomization=False, random_severity=0,
                    #                     maxmin_flag=False, remove_minus_flag=True, im_selection_mode=im, im_selection_extra=0.05,
                    #                     max_iter=None,
                    #                     )

    # methods = ['gradcam', 'fullcam', 'gradcampp', 'xgradcam']
    # maxmin = [False]
    # rmove = [True]
    # immode = ['all', 'diff_top']

    
    # for method in methods:
    #     for mm in maxmin:
    #         for rm in rmove:
    #             for im in immode:
    #                 evaluator_process(method=method, target_category=None, model_flag='vgg', task='Imagenet', dataset_split='val',
    #                                     randomization=False, random_severity=0,
    #                                     maxmin_flag=False, remove_minus_flag=True, im_selection_mode=im, im_selection_extra=0.05,
    #                                     max_iter=None, version=2,
    #                                     )




    # predictor_process(method='xgradcam', target_category=None, model_flag='vgg', task='Imagenet', dataset_split='val',
    #                     randomization=False, random_severity=0,
    #                     maxmin_flag=False, remove_minus_flag=True, im_selection_mode='diff_top', im_selection_extra=0.05,
    #                     max_iter=4800,
    #                     )


    # evaluator_process(method='gradcam', target_category=None, model_flag='vgg', task='Imagenet', dataset_split='val',
    #                     randomization=False, random_severity=0,
    #                     maxmin_flag=False, remove_minus_flag=True, im_selection_mode='all', im_selection_extra=0.05,
    #                     max_iter=None,
    #                     )