wget -O transcription.zip https://aidea-web.tw/file/bfcaa1b4-5b69-4f17-a5c4-f58ef7da68cb-1583372589_train_test_dataset_2___singingTranscription.zip
unzip transcription.zip
rm -r transcription.zip

mv singingTranscription data/singingTranscription

python3 lib/utils/download.py data/singingTranscription