DATASET_DIR=/nfs/data/kitti_raw/
DATASET_NAME=kitti_raw_eigen
DUMP_ROOT=/nfs/yue/my_kitti/
SEQ_LENGTH=3
IMG_HEIGHT=128
IMG_WIDTH=416
NUM_THREADS=16


python data/prepare_train_data.py --dataset_dir=${DATASET_DIR} --dataset_name=${DATASET_NAME} --dump_root=${DUMP_ROOT} --seq_length=${SEQ_LENGTH} --img_height=${IMG_HEIGHT} --img_width=${IMG_WIDTH} --num_threads=${NUM_THREADS} --remove_static
