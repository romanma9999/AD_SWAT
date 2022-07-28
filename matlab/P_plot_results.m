% plot results data
clc
close all
clear all

subsampling = 1; repeat_training = 1; anomalylikelihoodThreshold = 0.6;

load('swat_nominal.mat');
[P1n,P2n,P3n,P4n,P5n,P6n] = parse_swat(swat_nominal);
%  plot_swat(P1n,P2n,P3n,P4n,P5n,P6n);

load('swat_attack.mat');
[P1a,P2a,P3a,P4a,P5a,P6a] = parse_swat(swat_attack);
%plot_swat(P1a,P2a,P3a,P4a,P5a,P6a);

%results_file_name = '/P2_AIT202_learn_train_only_during_training_res.csv';
results_file_name = '/P1_P102_learn_train_only_freeze_off_res.csv';
startTime = '12/23/2015 15:00:00';
finishTrainingTime = datetime('28/12/2015 9:59:59','InputFormat','dd/MM/uuuu HH:mm:ss');

% first row is var idx
% second row is the location in the plot  (first, second etc ..)
% P_plot2

PID = 1;
P1 = P_preprocess(P1n, P1a, startTime,PID);
P1 = P_preprocess_results(results_file_name,P1,anomalylikelihoodThreshold,PID);
P1_AnomalyIdx = 6;
%P_plot2(P1,[1 6 10 7 8 9],P1_AnomalyIdx,PID); %LIT101
P_plot2(P1,[3 6 10 8 9 7],P1_AnomalyIdx,PID); %P102

% PID = 2;
% P2 = P_preprocess(P2n, P2a, startTime,PID);
% P2 = P_preprocess_results(results_file_name,P2,anomalylikelihoodThreshold,PID);
% P2_AnomalyIdx = 15;
% %P_plot2(P2,[4 15 19 16 17 18],P2_AnomalyIdx,PID); %FIT201
% P_plot2(P2,[2 15 19 16 17 18],P2_AnomalyIdx,PID); %AIT202
% P_plot2(P2,[8 15 19 16 17 18],P2_AnomalyIdx,PID); %P203

