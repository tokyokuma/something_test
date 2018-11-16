#!/bin/sh
echo "start"
python all_to_extract_rgb_label.py
echo "finish extract"
python ~/something_test/python/finish_mp3.py

python extract_rgb_gray_and_rename.py
echo "finish rename"
python ~/something_test/python/finish_mp3.py

python resize.py
echo "finish resize"
python ~/something_test/python/finish_mp3.py
