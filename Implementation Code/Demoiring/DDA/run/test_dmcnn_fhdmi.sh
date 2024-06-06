test_path=""
clean_path=""
dataset_name=custom
arch=DMCNN
export CUDA_VISIBLE_DEVICES="0"

test(){
    python main.py \
    --arch DMCNN \
    --testdata_path $test_path \
    --cleandata_path $clean_path \
    --dataset $dataset_name \
    --moire_type Real \
    --Test_pretrained_path './ckpt/Best_performance_MCNN_class_statedict_epoch144_psnr35.421.pth' \
    --batchsize 1 \
    --tensorboard \
    --width_list 0.75 0.5 0.25 \
    --operation test \
    --name "test"
}

test
