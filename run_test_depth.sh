MODE=test_depth
DATASET_DIR=/nfs/data/kitti_raw/
INIT_CKPT_FILE=/nfs/yue/GeoNet/models/geonet_depthnet/model
BATCH_SIZE=1
DEPTH_TEST_SPLIT=eigen
OUTPUT_DIR=/nfs/yue/GeoNet/outputs/depth/

python geonet_main.py --mode=${MODE} --dataset_dir=${DATASET_DIR} --init_ckpt_file=${INIT_CKPT_FILE} --batch_size=${BATCH_SIZE} --depth_test_split=${DEPTH_TEST_SPLIT} --output_dir=${OUTPUT_DIR}
