python3 train.py \
    train.json val.json \
    --model_config config/model5.json \
    --lr 0.0001 \
    --epochs 150 \
    --weight_decay 0.00005 \
    --dropout 0 \
    --criterion bceloss \
    --batch_size 32 \
    --name maxpool_cnn_rnn3_model5_bidirectional
