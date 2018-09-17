SPLIT=eigen
KITTI_DIR=/nfs/data/kitti_raw/
PRED_FILE=/nfs/yue/GeoNet/outputs/depth/model.npy


python kitti_eval/eval_depth.py --split=${SPLIT} --kitti_dir=${KITTI_DIR}  --pred_file=${PRED_FILE}
#python geonet_main.py --mode=${MODE} --dataset_dir=${DATASET_DIR} --init_ckpt_file=${INIT_CKPT_FILE} --batch_size=${BATCH_SIZE} --depth_test_split=${DEPTH_TEST_SPLIT} --output_dir=${OUTPUT_DIR}
