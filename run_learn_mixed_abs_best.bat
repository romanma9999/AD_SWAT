python swat_htm.py --verbose -sn P1 -cn P102 -ctype 1 -ft off -lt always --sdr_size 60
python swat_htm.py --verbose -sn P1 -cn LIT101 -ctype 0 -ft off -lt always -sbp -w 5 -size 1024
python calc_anomaly_stats.py -sn P1 -esn P1 -ft off -lt always -ofa _learn_mixed -bcn LIT101,P102 

python swat_htm.py --verbose -sn P2 -cn AIT202 -ctype 0 -ft off -lt always -sbp -w 34 -size 2048
python swat_htm.py --verbose -sn P2 -cn AIT203 -ctype 0 -ft off -lt always 
python swat_htm.py --verbose -sn P2 -cn FIT201 -ctype 0 -ft off -lt always -sbp -w 13 -size 1024
python swat_htm.py --verbose -sn P2 -cn MV201 -ctype 1 -ft off -lt train_only --sdr_size 60
python swat_htm.py --verbose -sn P2 -cn P201 -ctype 1 -ft off -lt always --sdr_size 60
python swat_htm.py --verbose -sn P2 -cn P203 -ctype 1 -ft off -lt train_only --sdr_size 60
python swat_htm.py --verbose -sn P2 -cn P204 -ctype 1 -ft off -lt train_only --sdr_size 60
python swat_htm.py --verbose -sn P2 -cn P205 -ctype 1 -ft off -lt train_only --sdr_size 60
python swat_htm.py --verbose -sn P2 -cn P206 -ctype 1 -ft off -lt train_only --sdr_size 60
python calc_anomaly_stats.py -sn P2 -esn P2 -ft off -lt always -ofa _learn_mixed -bcn AIT202,FIT201,MV201,P201,P203,P204,P205

python swat_htm.py --verbose -sn P3 -cn DPIT301 -ctype 0 -ft off -lt always -sbp -w 21 -size 512
python swat_htm.py --verbose -sn P3 -cn FIT301 -ctype 0 -ft off -lt always -sbp -w 34 -size 512
python swat_htm.py --verbose -sn P3 -cn LIT301 -ctype 0 -ft off -lt always -sbp -w 1 -size 512
python swat_htm.py --verbose -sn P3 -cn MV301 -ctype 1 -ft off -lt train_only --sdr_size 60
python swat_htm.py --verbose -sn P3 -cn MV302 -ctype 1 -ft off -lt train_only --sdr_size 60
python swat_htm.py --verbose -sn P3 -cn MV303 -ctype 1 -ft off -lt train_only --sdr_size 60
python swat_htm.py --verbose -sn P3 -cn MV304 -ctype 1 -ft off -lt train_only --sdr_size 60
python swat_htm.py --verbose -sn P3 -cn P302 -ctype 1 -ft off -lt train_only --sdr_size 60
python calc_anomaly_stats.py -sn P3 -esn P3 -ft off -lt always -ofa _learn_mixed -bcn DPIT301,FIT301,LIT301,MV301,MV302,MV303,MV304,P302

python swat_htm.py --verbose -sn P4 -cn AIT402 -ctype 0 -ft off -lt always -sbp -w 1 -size 256
python swat_htm.py --verbose -sn P4 -cn FIT401 -ctype 0 -ft off -lt always -sbp -w 21 -size 1024
python swat_htm.py --verbose -sn P4 -cn LIT401 -ctype 0 -ft off -lt always -sbp -w 1 -size 512
python swat_htm.py --verbose -sn P4 -cn P402 -ctype 1 -ft off -lt train_only --sdr_size 60
python swat_htm.py --verbose -sn P4 -cn P403 -ctype 1 -ft off -lt train_only --sdr_size 60
python calc_anomaly_stats.py -sn P4 -esn P4 -ft off -lt always -ofa _learn_mixed -bcn AIT402,FIT401,LIT401,P402,P403 

python swat_htm.py --verbose -sn P5 -cn AIT501 -ctype 0 -ft off -lt always -sbp -w 1 -size 256
python swat_htm.py --verbose -sn P5 -cn AIT502 -ctype 0 -ft off -lt always -sbp -w 1 -size 256
python swat_htm.py --verbose -sn P5 -cn AIT503 -ctype 0 -ft off -lt always -sbp -w 34 -size 512
python swat_htm.py --verbose -sn P5 -cn AIT504 -ctype 0 -ft off -lt always -sbp -w 21 -size 1024
python swat_htm.py --verbose -sn P5 -cn FIT501 -ctype 0 -ft off -lt always -sbp -w 21 -size 1024
python swat_htm.py --verbose -sn P5 -cn FIT502 -ctype 0 -ft off -lt always -sbp -w 1 -size 256
python swat_htm.py --verbose -sn P5 -cn FIT503 -ctype 0 -ft off -lt always -sbp -w 21 -size 2048
python swat_htm.py --verbose -sn P5 -cn FIT504 -ctype 0 -ft off -lt always -sbp -w 21 -size 2048
python swat_htm.py --verbose -sn P5 -cn P501 -ctype 1 -ft off -lt train_only --sdr_size 60
python swat_htm.py --verbose -sn P5 -cn PIT501 -ctype 0 -ft off -lt always -sbp -w 5 -size 1024
:: python swat_htm.py --verbose -sn P5 -cn PIT502 -ctype 0 -ft off -lt always 
python swat_htm.py --verbose -sn P5 -cn PIT503 -ctype 0 -ft off -lt always -sbp -w 8 -size 1024
python calc_anomaly_stats.py -sn P5 -esn P5 -ft off -lt always -ofa _learn_mixed -bcn AIT501,AIT502,AIT503,AIT504,FIT501,FIT502,FIT503,FIT504,P501,PIT501,PIT503

python swat_htm.py --verbose -sn P6 -cn FIT601 -ctype 0 -ft off -lt always -sbp -w 34 -size 512
python swat_htm.py --verbose -sn P6 -cn P602 -ctype 1 -ft off -lt train_only --sdr_size 60
python calc_anomaly_stats.py -sn P6 -esn P6 -ft off -lt always -ofa _learn_mixed -bcn FIT601,P602 
python calc_anomaly_stats.py -ofa _learn_mixed --final_stage 