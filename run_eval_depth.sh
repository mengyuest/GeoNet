SPLIT=eigen
KITTI_DIR=/nfs/data/kitti_raw/
PRED_FILE=/nfs/yue/GeoNet/outputs/depth/model.npy
VIS_DIR=/nfs/yue/GeoNet/outputs/depth/vis/

#python kitti_eval/eval_depth.py --split=${SPLIT} --kitti_dir=${KITTI_DIR}  --pred_file=${PRED_FILE}
python kitti_eval/eval_depth_vis.py --split=${SPLIT}  --kitti_dir=${KITTI_DIR} --pred_file=${PRED_FILE} --vis_dir=${VIS_DIR}
