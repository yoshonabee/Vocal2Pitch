python3 train.py \
    train.json val.json \
    --model_config config/model5.json \
    --weight_decay 0.001 \
    --dropout 0 \
    --criterion bceloss \
    --name cnn_rnn3_model5_bidirectional
