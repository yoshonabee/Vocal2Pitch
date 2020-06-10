python3 train.py \
    train.json val.json \
    --model_config config/model5.json \
    --lr 0.001 \
    --epochs 150 \
    --dropout 0 \
    --criterion bceloss \
    --batch_size 128 \
    --name fix_dataAug5_cnn_model5_transformer_WD0_LR1e-3
