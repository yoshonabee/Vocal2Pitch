python3 train.py \
    train.json val.json \
    --model_config config/model5.json \
    --lr 0.001 \
    --weight_decay 0 \
    --device cpu \
    --dropout 0 \
    --criterion bceloss \
    --name test
