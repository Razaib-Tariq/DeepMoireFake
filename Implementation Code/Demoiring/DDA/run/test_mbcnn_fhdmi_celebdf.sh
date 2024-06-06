test_path=CelebDF
clean_path=/CelebDF/OG/Real # this path can be anything it is just for args.
dataset_name=custom # custom function to demoire the moire images.
#dataset_name=custom_mouth for lipforensics
arch=MBCNN
export CUDA_VISIBLE_DEVICES="3"

test(){
    python main_celebdf.py \
    --arch MBCNN \
    --testdata_path $test_path \
    --cleandata_path $clean_path \
    --dataset $dataset_name \
    --moire_type Real \
    --Test_pretrained_path 'DDA/ckpt/MBCNN_fhdmi.pth' \
    --batchsize 1 \
    --tensorboard \
    --width_list 0.75 0.5 0.25 \
    --operation test \
    --name "test"
}

test


