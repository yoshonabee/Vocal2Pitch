python3 train.py \
    train.json val.json \
    --model_config config/16ms_model5.json \
    --lr 0.0001 \
    --epochs 150 \
    --dropout 0 \
    --criterion bceloss \
    --batch_size 256 \
    --inbalance_ratio 5 \
    --name 16ms_resample5_fix_dataAug5_cnn_16msModel5_transformer_WD0_LR1e-4
