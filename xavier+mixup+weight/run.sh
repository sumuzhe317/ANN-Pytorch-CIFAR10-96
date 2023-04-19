#!/bin/bash
python main.py -g 1 -e 200 --decay 5e-4
bash ../../../../send_msg_rebot.sh Done
