python3 train.py \
    train.json val.json \
    --model_config config/model5.json \
    --lr 0.0001 \
    --weight_decay 0 \
    --dropout 0 \
    --batch_size 128 \
    --criterion bceloss \
    --name official_feature_only_model5
