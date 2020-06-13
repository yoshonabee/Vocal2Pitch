python3 train.py \
    train.json val.json \
    --model_config config/model5.json \
    --lr 0.001 \
    --weight_decay 0 \
    --dropout 0 \
    --criterion bceloss \
    --name 20ms_crepe_as_input
