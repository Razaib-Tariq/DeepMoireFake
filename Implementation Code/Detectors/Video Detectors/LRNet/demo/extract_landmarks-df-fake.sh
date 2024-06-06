#!/bin/bash

CUDA_VISIBLE_DEVICES="3" python extract_landmarks-df.py \
    -i /media/data1/razaib/Moire/OG/FF++/deepfakes/Deepfakes-fake_10seconds \
    -o /media/data1/razaib/Moire/OG/FF++/deepfakes/landmarks_new/Deepfakes-fake_10seconds/landmarks \
    --fd blazeface