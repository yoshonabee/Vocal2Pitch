python3 train.py \
    train.json val.json \
    --model_config config/model5.json \
    --weight_decay 0.001 \
    --dropout 0 \
    --criterion bceloss \
    --name spleeter_cnn_32ms_lr1e-3_batchnorm_bceloss_model5_score1_0_dropout0.5
