from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import torch


def stat_calculator(cam_grad_max_matrix, cam_grad_min_matrix):
    grad_max_percentile = np.percentile(cam_grad_max_matrix, (25, 50, 75, 90, 99))
    grad_min_percentile = np.percentile(cam_grad_min_matrix, (25, 50, 75, 90, 99))
    return grad_max_percentile, grad_min_percentile


def cam_stats_step(cam_algorithm, target_layers, general_args,# for the cam setting
                    model, target_dataset, num_out_channel:int, num_classes:int=2,  # for the model and dataset
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # use None as the target category
    test_loader = DataLoader(target_dataset, batch_size=general_args.batch_size, shuffle=False,
                                num_workers=1, pin_memory=True)
    in_fold_counter = 0
    in_fold_target_counter = np.zeros([num_classes], dtype=np.int16)
    # TODO improve the num_out_channel calculation
    im_overall = np.zeros([num_out_channel], dtype=np.float32)  # 400 = number of channels per group
    im_target = np.zeros([num_classes, num_out_channel], dtype=np.float32)
    im_diff = np.zeros([num_classes, num_out_channel], dtype=np.float32)
    # cammax & cammin 用于进行总体的maxmin 归一化
    cam_grad_max_matrix = []
    cam_grad_min_matrix = []
    
    model = model.to(device=device)
    model.eval()
    for x,y in tqdm(test_loader):
        x = x.to(dtype=torch.float32).to(device)
        y = y.to(dtype=torch.float32).to(device)
        with cam_algorithm(model=model,
                            target_layers=target_layers,
                            use_cuda=True) as cam:
            grayscale_cam, predict_category, confidence, cam_grad_max_value, cam_grad_min_value\
                                                                                    = cam(input_tensor=x, 
                                                                                        target_category=general_args.target_category_flag,
                                                                                        )
            # cam_single_max_value - [batch, 1*value]已经展平 --[batch]
            cam_grad_max_matrix.extend(cam_grad_max_value)
            cam_grad_min_matrix.extend(cam_grad_min_value)

            # proved: grayscale_cam - 1* [16, 512]
            grayscale_cam = grayscale_cam[0]  # [1, all_channel] remove the target layers
            # grayscale_cam - [16, 512]
            for i in range(general_args.batch_size): # [all_channel]
                single_grayscale_cam, single_predict_category, single_confidence = grayscale_cam[i], predict_category[i], confidence[i]
                if general_args.confidence_weight_flag:
                    single_grayscale_cam = single_grayscale_cam * single_confidence
                    single_max_reviser = single_confidence
                else:
                    single_max_reviser = 1

                if single_predict_category == y.data.cpu().numpy()[i]:  # 只叠加正确分类的部分
                    # 添加总体IM
                    im_overall = im_overall + single_grayscale_cam
                    in_fold_counter += single_max_reviser
                    # 添加对应类的IM
                    
                    im_target[single_predict_category] = im_target[single_predict_category] + single_grayscale_cam
                    in_fold_target_counter[single_predict_category] += single_max_reviser
    # im_target - [num_classes, num_features]
    im_overall = im_overall / in_fold_counter
    im_target = im_target / in_fold_target_counter[:, None]
    for i in range(num_classes):
        im_diff[i] = im_target[i, :] - im_overall

    # im_overall [num_out_channel]
    # im_target/im_diff [num_classes, num_out_channel]
    # 此处im不分group因为不同group的feature在heatmap上就应该不同，在重要性上的差异也应该保留
    # 而max min不分group或者类别因为需要全局统一尺度，无论group或者batch或者category
    # 计算分位数
    cam_grad_max_matrix = np.array(cam_grad_max_matrix)
    cam_grad_min_matrix = np.array(cam_grad_min_matrix)
    cam_grad_max_matrix, cam_grad_min_matrix = stat_calculator(cam_grad_max_matrix, cam_grad_min_matrix)
    return im_overall, im_target, im_diff, cam_grad_max_matrix, cam_grad_min_matrix