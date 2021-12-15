% plot results data
clc
close all
clear all

subsampling = 1; repeat_training = 1; anomalylikelihoodThreshold = 0.1;

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
P1 = P_preprocess_results(P1,anomalylikelihoodThreshold,PID);
AnomalyIdx = 5;
%P_plot2(P1,[1 5 8;1 3 2],AnomalyIdx,PID);
P_plot2(P1,[1 5 8 9;1 4 2 3],AnomalyIdx,PID);

