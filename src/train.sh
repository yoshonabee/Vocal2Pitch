python3 train.py \
    train.json val.json \
    --model_config config/model5.json \
    --feature_config config/feature_time.json \
    --weight_decay 0 \
    --batch_size 32 \
    --name spleeter_cnn_32ms_val100_batchsize32_featureTime_batchnorm_crossentropy_bias_model5
