python3 train.py \
    train.json val.json \
    --model_config config/small_model5.json \
    --weight_decay 0.0002 \
    --dropout 0 \
    --criterion bceloss \
    --name cnn_rnn3_small_model5_bidirectional
