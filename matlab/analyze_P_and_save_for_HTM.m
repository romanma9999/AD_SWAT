clc
clear all
close all
%swat_nominal = load_swat('../swat/SWaT_Dataset_Normal_v1.xlsx');
%swat_attack = load_swat('../swat/SWaT_Dataset_Attack_v0.xlsx');
%save('swat_nominal.mat','swat_nominal');
%save('swat_attack.mat','swat_attack');

vars = {'swat_nominal','swat_attack','Pa','Pn'};

load('swat_nominal.mat');
[Pn{1},Pn{2},Pn{3},Pn{4},Pn{5},Pn{6}] = parse_swat(swat_nominal);
%plot_swat(Pn{1},Pn{2},Pn{3},Pn{4},Pn{5},Pn{6});

load('swat_attack.mat');
[Pa{1},Pa{2},Pa{3},Pa{4},Pa{5},Pa{6}] = parse_swat(swat_attack);
%plot_swat(Pa{1},Pa{2},Pa{3},Pa{4},Pa{5},Pa{6});


startTime = '12/23/2015 15:00:00';
finishTrainingTime = datetime('28/12/2015 9:59:59','InputFormat','dd/MM/uuuu HH:mm:ss');
for i = 1:6
    p_prepare_for_HTM(Pn{i},Pa{i},startTime, finishTrainingTime,i)
end
% i = 2
% p_prepare_for_HTM(Pn{i},Pa{i},startTime, finishTrainingTime,i)
clear(vars{:})




