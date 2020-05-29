python3 train.py \
    train.json val.json \
    --model_config config/model5.json \
    --weight_decay 0 \
    --segment_length 4 \
    --dropout 0 \
    --batch_size 32 \
    --name spleeter_cnn_32ms_val100_batchsize32_segmentLen4_batchnorm_crossentropy_bias_model5
