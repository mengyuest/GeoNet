# Mostly based on the code written by Clement Godard: 
# https://github.com/mrharicot/monodepth/blob/master/utils/evaluate_kitti.py
from __future__ import division
import sys
import cv2
import os
import numpy as np
import argparse
from depth_evaluation_utils import *

#TODO to make plt work in non X-server env
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str, default='eigen', help='eigen or stereo split')
parser.add_argument("--kitti_dir", type=str, help='Path to the KITTI dataset directory')
parser.add_argument("--pred_file", type=str, help="Path to the prediction file")
parser.add_argument('--min_depth', type=float, default=1e-3, help="Threshold for minimum depth")
parser.add_argument('--max_depth', type=float, default=80, help="Threshold for maximum depth")
parser.add_argument('--vis_dir', type=str, help='Path to the visualization output directory')
args = parser.parse_args()

with open('data/kitti/test_files_%s.txt' % args.split, 'r') as f:
    ori_test_files = f.readlines()
    ori_test_files = [args.kitti_dir + t[:-1] for t in ori_test_files]
#print(ori_test_files[:10])
def convert_disps_to_depths_stereo(gt_disparities, pred_depths):
    gt_depths = []
    pred_depths_resized = []
    pred_disparities_resized = []
    
    for i in range(len(gt_disparities)):
        gt_disp = gt_disparities[i]
        height, width = gt_disp.shape

        pred_depth = pred_depths[i]
        pred_depth = cv2.resize(pred_depth, (width, height), interpolation=cv2.INTER_LINEAR)

        pred_disparities_resized.append(1./pred_depth) 

        mask = gt_disp > 0

        gt_depth = width_to_focal[width] * 0.54 / (gt_disp + (1.0 - mask))
        #pred_depth = width_to_focal[width] * 0.54 / pred_disp

        gt_depths.append(gt_depth)
        pred_depths_resized.append(pred_depth)
    return gt_depths, pred_depths_resized, pred_disparities_resized

def main():

    pred_depths = np.load(args.pred_file)
    args.test_file_list = './data/kitti/test_files_%s.txt' % args.split

    print('evaluating ' + args.pred_file + '...') # adapt to py3

    if args.split == 'eigen':
        test_files = read_text_lines(args.test_file_list)
        gt_files, gt_calib, im_sizes, im_files, cams = \
            read_file_data(test_files, args.kitti_dir)
        num_test = 10 #TODO len(im_files)
        gt_depths = []
        pred_depths_resized = []
        for t_id in range(num_test):
            camera_id = cams[t_id]  # 2 is left, 3 is right
            pred_depths_resized.append(
                cv2.resize(pred_depths[t_id], 
                           (im_sizes[t_id][1], im_sizes[t_id][0]), 
                           interpolation=cv2.INTER_LINEAR))
            depth = generate_depth_map(gt_calib[t_id], 
                                       gt_files[t_id], 
                                       im_sizes[t_id], 
                                       camera_id, 
                                       False, 
                                       True)
            gt_depths.append(depth.astype(np.float32))
        pred_depths = pred_depths_resized
    else:
        num_test = 200
        gt_disparities = load_gt_disp_kitti(args.kitti_dir)
        gt_depths, pred_depths, pred_disparities_resized = convert_disps_to_depths_stereo(gt_disparities, pred_depths)

    rms     = np.zeros(num_test, np.float32)
    log_rms = np.zeros(num_test, np.float32)
    abs_rel = np.zeros(num_test, np.float32)
    sq_rel  = np.zeros(num_test, np.float32)
    d1_all  = np.zeros(num_test, np.float32)
    a1      = np.zeros(num_test, np.float32)
    a2      = np.zeros(num_test, np.float32)
    a3      = np.zeros(num_test, np.float32)
    for i in range(num_test):    
        gt_depth = gt_depths[i]
        pred_depth = np.copy(pred_depths[i])

        if args.split == 'eigen':

            mask = np.logical_and(gt_depth > args.min_depth, 
                                  gt_depth < args.max_depth)
            # crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
            # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
            gt_height, gt_width = gt_depth.shape
            crop = np.array([0.40810811 * gt_height,  0.99189189 * gt_height,   
                             0.03594771 * gt_width,   0.96405229 * gt_width]).astype(np.int32)

            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        if args.split == 'stereo':
            gt_disp = gt_disparities[i]
            mask = gt_disp > 0
            pred_disp = pred_disparities_resized[i]

            disp_diff = np.abs(gt_disp[mask] - pred_disp[mask])
            bad_pixels = np.logical_and(disp_diff >= 3, (disp_diff / gt_disp[mask]) >= 0.05)
            d1_all[i] = 100.0 * bad_pixels.sum() / mask.sum()

        # Scale matching
        scalor = np.median(gt_depth[mask])/np.median(pred_depth[mask])
        pred_depth[mask] *= scalor

        pred_depth[pred_depth < args.min_depth] = args.min_depth
        pred_depth[pred_depth > args.max_depth] = args.max_depth
        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = \
            compute_errors(gt_depth[mask], pred_depth[mask])
        
        # visualization
        img_gt=gt_depth
        img_pred=pred_depth
        img_abs_rel=np.zeros(np.shape(gt_depth))
        img_se=np.zeros(np.shape(gt_depth))
        img_se_log=np.zeros(np.shape(gt_depth))
        img_abs_rel=np.abs(img_gt - img_pred) / img_gt
        img_se = (img_gt - img_pred) ** 2
        img_se_log = (np.log(img_gt) - np.log(img_pred)) ** 2
        img_h, img_w = np.shape(gt_depth)
        origin_img=plt.imread(ori_test_files[i])
        plt.imshow(origin_img)
        plt.savefig(args.vis_dir+"%04d_origin.png"%(i))
        plt.close()
        plt.imshow(mask)
        plt.savefig(args.vis_dir+"%04d_mask.png"%(i))
        plt.close()
        plt.imshow(gt_depth)
        plt.savefig(args.vis_dir+"%04d_gt.png"%(i))
        plt.close()
        plt.imshow(pred_depth)
        plt.savefig(args.vis_dir+"%04d_pred.png"%(i))
        plt.close()
        plt.imshow(img_abs_rel)
        plt.savefig(args.vis_dir+"%04d_absrel.png"%(i))
        plt.close()
        plt.imshow(img_se)
        plt.savefig(args.vis_dir+"%04d_se.png"%(i))
        plt.close()
        plt.imshow(img_se_log)
        plt.savefig(args.vis_dir+"%04d_se_log.png"%(i))
        plt.close()        
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3'))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean(), a2.mean(), a3.mean()))    

main()
