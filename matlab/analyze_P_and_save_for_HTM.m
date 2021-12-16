clc
clear all
close all
%swat_nominal = load_swat('../swat/SWaT_Dataset_Normal_v1.xlsx');
%swat_attack = load_swat('../swat/SWaT_Dataset_Attack_v0.xlsx');
%save('swat_nominal.mat','swat_nominal');
%save('swat_attack.mat','swat_attack');

vars = {'sensors_data','swat_nominal','swat_attack','P1T','P1n','P2n','P3n','P4n','P5n','P6n','P1a','P2a','P3a','P4a','P5a','P6a'};

load('swat_nominal.mat');
[P1n,P2n,P3n,P4n,P5n,P6n] = parse_swat(swat_nominal);
%  plot_swat(P1n,P2n,P3n,P4n,P5n,P6n);

load('swat_attack.mat');
[P1a,P2a,P3a,P4a,P5a,P6a] = parse_swat(swat_attack);
%plot_swat(P1a,P2a,P3a,P4a,P5a,P6a);

PID = 1;
startTime = '12/23/2015 14:00:00';
finishTrainingTime = datetime('28/12/2015 9:59:59','InputFormat','dd/MM/uuuu HH:mm:ss');

P1 = P_preprocess(P1n, P1a, startTime,PID);
P_plot(P1,[1 2 3 4 5],PID);

TrainSamplesCount = find(P1.Time == finishTrainingTime);
disp(['train samples count is ' num2str(TrainSamplesCount)]);

sensors_data = P1.Variables;
n_all_data = size(sensors_data,2)-1; %remove 'Anomaly' label


sensors_data = sensors_data(:,1:n_all_data);

writematrix(sensors_data,['../HTM_input/P1_data.csv']) 

meta_data = zeros(1,1+2*n_all_data);
meta_data(1) = TrainSamplesCount;
meta_data(2:n_all_data+1) = floor(min(sensors_data));
meta_data(n_all_data+2:2*n_all_data+1) = ceil(max(sensors_data));
writematrix(meta_data,['../HTM_input/P1_meta.csv']) 

clear(vars{:})



