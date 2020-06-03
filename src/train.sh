python3 train.py \
    train.json val.json \
    --model_config config/model14.json \
    --feature_config config/feature4.json \
    --weight_decay 0 \
    --epochs 150 \
    --lr 0.0001 \
    --batch_size 1024 \
    --name framehop1_width31_residual3_mel120_normalize_all
