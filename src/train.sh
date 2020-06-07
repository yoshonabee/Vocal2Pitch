python3 train.py \
    train.json val.json \
    --model_config config/model5.json \
    --lr 0.001 \
    --weight_decay 0 \
    --dropout 0 \
    --criterion bceloss \
    --name dropout0.5_feature_cnn_rnn3_model5_bidirectional_no_weight_decay
