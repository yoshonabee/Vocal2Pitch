python3 train.py \
    train.json val.json \
    --model_config config/model14.json \
    --feature_config config/feature4.json \
    --weight_decay 0 \
    --lr 0.0001 \
    --batch_size 1024 \
    --name framehop1_width15_2layer
