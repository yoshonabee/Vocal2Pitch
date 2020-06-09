python3 train.py \
    train.json val.json \
    --model_config config/model5.json \
    --lr 0.001 \
    --epochs 150 \
    --weight_decay 0.00005 \
    --dropout 0 \
    --criterion bceloss \
    --batch_size 128 \
    --name cnn_model5_transformer_WD5e-5_LR1e-3
