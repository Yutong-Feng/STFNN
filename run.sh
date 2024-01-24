nohup \
python -u train.py --hid_dim 16 > /dev/null 2>&1 && \
python -u train.py --hid_dim 32 > /dev/null 2>&1 && \
python -u train.py --hid_dim 48 > /dev/null 2>&1 && \
python -u train.py --dim_feed 16 > /dev/null 2>&1 && \
python -u train.py --dim_feed 32 > /dev/null 2>&1 && \
python -u train.py --dim_feed 48 > /dev/null 2>&1 &