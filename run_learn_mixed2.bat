python swat_htm.py --verbose -sn P2 -cn AIT202 -ctype 0 -ft during_training -lt always 
python swat_htm.py --verbose -sn P2 -cn FIT201 -ctype 0 -ft during_training -lt always 
python swat_htm.py --verbose -sn P2 -cn MV201 -ctype 1 -ft off -lt train_only --sdr_size 60
python swat_htm.py --verbose -sn P2 -cn P203 -ctype 1 -ft off -lt train_only --sdr_size 60
python swat_htm.py --verbose -sn P2 -cn P204 -ctype 1 -ft off -lt train_only --sdr_size 60
python swat_htm.py --verbose -sn P2 -cn P205 -ctype 1 -ft off -lt train_only --sdr_size 60
python swat_htm.py --verbose -sn P2 -cn P206 -ctype 1 -ft off -lt train_only --sdr_size 60
python calc_anomaly_stats.py -sn P2 -esn P2 -ft off -lt always -ofa _learn_mixed2 -bcn AIT202,FIT201,MV201,P203,P204,P205

