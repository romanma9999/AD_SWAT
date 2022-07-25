python P1_LIT101.py --verbose -sn P1 -cn LIT101 -ft off -lt always 
python P1_LIT101.py --verbose -sn P1 -cn P101 -ft off -lt always 
python P1_LIT101.py --verbose -sn P1 -cn P102 -ft off -lt always 
python P1_LIT101.py --verbose -sn P1 -cn MV101 -ft off -lt always 
python calc_anomaly_stats.py -sn P1 -bcn LIT101,P101,P102,MV101 -ft off -lt always 