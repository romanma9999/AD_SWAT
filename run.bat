python swat_htm.py --verbose -sn P1 -cn LIT101 -ctype 0 -ft off -lt always 
python swat_htm.py --verbose -sn P1 -cn P102 -ctype 1 -ft off -lt train_only --sdr_size 60
python calc_anomaly_stats.py -sn P1 -ft off -lt always  -bcn LIT101,P102 -bfn P1_LIT101_learn_always_freeze_off,P1_P102_learn_train_only_freeze_off