python3 train.py \
    train.json val.json \
    --model_config config/large_model5.json \
    --lr 0.0001 \
    --weight_decay 0.0005 \
    --dropout 0 \
    --criterion bceloss \
    --name large_featureNorm_do0.5_cnn_rnn3_model5_bidirectional_wd2e-4_lr1e-4
