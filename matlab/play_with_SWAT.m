
clc
clear all
close all

load('swat_nominal.mat');
[Pn{1},Pn{2},Pn{3},Pn{4},Pn{5},Pn{6}] = parse_swat(swat_nominal);
%plot_swat(Pn{1},Pn{2},Pn{3},Pn{4},Pn{5},Pn{6});

%idx = [6,9,11,13];
load('swat_attack.mat');
[Pa{1},Pa{2},Pa{3},Pa{4},Pa{5},Pa{6}] = parse_swat(swat_attack);
plot_swat(Pa{1},Pa{2},Pa{3},Pa{4},Pa{5},Pa{6});
%PA2 = addvars(Pa{2},Pa{2}{:,12},'After','Normal','NewVariableNames','Attack');
%plot_swat(Pa{1},PA2(:,idx),Pa{3},Pa{4},Pa{5},Pa{6});

PID = 2

% unite training and attack parts of the dataset
PT = [Pn{PID}; Pa{PID}];
PT.Normal(:) = 0;

startTime = '12/23/2015 15:00:00';
finishTrainingTime = datetime('28/12/2015 9:59:59','InputFormat','dd/MM/uuuu HH:mm:ss');

% crop beginning
TR = timerange(startTime,'inf');
P =  PT(TR,:);



% mark the anomalies relevant to PID
[PA, PA_idx] = get_anomaly_times(PID);

nPA = length(PA_idx);
P.Properties.VariableNames{end} = 'Attack';

for j = 1:nPA
    dstart = datetime(PA(PA_idx(j)).s,'InputFormat','MM/dd/uuuu HH:mm:ss');
    dend = datetime(PA(PA_idx(j)).e,'InputFormat','MM/dd/uuuu HH:mm:ss');
    TR = timerange(dstart,dend);
    P(TR,:).Attack(:) = 1;
end


val = movmean(P.FIT201,[14 0]);
d = diff(val);
maxx = max(d);
minn = min(d);
diff_norm = (d - minn)/(maxx - minn);
diff_norm = diff_norm - mean(diff_norm);
diff_ang = atan(diff_norm*100)*180/pi;

figure;
plot(val);
title('FIT201')
figure;
plot(d)
title('diff')
figure;
plot(diff_norm*100)
title('diff norm')
figure;
plot(diff_ang)
title('diff ang')
figure;
histogram(val,'Normalization','probability')
title('histogram FIT201')
figure;
histogram(diff_norm*100,'Normalization','probability')
title('histogram diff')
figure;
histogram(diff_ang,90,'Normalization','probability')
title('diff ang')
