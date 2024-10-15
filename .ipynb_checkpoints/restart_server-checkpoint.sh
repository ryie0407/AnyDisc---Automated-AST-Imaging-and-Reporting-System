#!/bin/bash
export PATH="/home/ubuntu/miniconda3/bin:$PATH"
export PYTHONPATH="/home/ubuntu/miniconda3/lib/python3.10/site-packages"

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate base

if ! pgrep -f "server.py" > /dev/null; then
    echo "$(date): Restarting server.py" >> /home/ubuntu/02.WGS/01.AnyDisc/server.log
    cd /home/ubuntu/02.WGS/01.AnyDisc
    nohup python server.py &>> server.log &
else
    echo "$(date): server.py is already running." >> /home/ubuntu/02.WGS/01.AnyDisc/server.log
fi
