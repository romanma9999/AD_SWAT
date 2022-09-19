% plot results data
clc
close all
clear all

subsampling = 20; repeat_training = 1; anomalylikelihoodThreshold = 0.6;


load('swat_nominal.mat');
[P1n,P2n,P3n,P4n,P5n,P6n] = parse_swat(swat_nominal,subsampling);
%  plot_swat(P1n,P2n,P3n,P4n,P5n,P6n);

load('swat_attack.mat');
[P1a,P2a,P3a,P4a,P5a,P6a] = parse_swat(swat_attack,subsampling);
%plot_swat(P1a,P2a,P3a,P4a,P5a,P6a);


results_file_name = '/P4_AIT402_learn_always_freeze_off_res.csv';
%results_file_name = '/P6_FIT601_learn_always_freeze_off_res.csv';
%results_file_name = '/P2_AIT202_learn_always_freeze_off_res.csv';
%results_file_name = '/P5_FIT502_learn_always_freeze_off_res.csv';
%results_file_name = '/P3_DPIT301_learn_always_freeze_off_res.csv';
%results_file_name = '/P1_LIT101_learn_always_freeze_off_res.csv';
%results_file_name = '/P1_LIT101_learn_train_only_freeze_off_res.csv';
%results_file_name = '/P1_LIT101_learn_always_freeze_off_res.csv';
%results_file_name = '/P2_FIT201_learn_always_freeze_off_res.csv';
startTime = '12/23/2015 15:00:00';
finishTrainingTime = datetime('28/12/2015 9:59:59','InputFormat','dd/MM/uuuu HH:mm:ss');

% first row is var idx
% second row is the location in the plot  (first, second etc ..)
% P_plot2

% PID = 1;
% P1 = P_preprocess(P1n, P1a, startTime,PID);
% P1 = P_preprocess_results(results_file_name,P1,anomalylikelihoodThreshold,PID);
% P1_AnomalyIdx = 6;
% P_plot2(P1,[1 11 6 10 7 8 9],P1_AnomalyIdx,PID); %LIT101
% %%P_plot2(P1,[4 6 10 8 9 7],P1_AnomalyIdx,PID); %P102
% 
% PID = 2;
% P2 = P_preprocess(P2n, P2a, startTime,PID);
% P2 = P_preprocess_results(results_file_name,P2,anomalylikelihoodThreshold,PID);
% P2_AnomalyIdx = 13;
% %P_plot2(P2,[4 18 13 17 15 16],P2_AnomalyIdx,PID); %FIT201
% P_plot2(P2,[2 18 13 17 15],P2_AnomalyIdx,PID); %AIT202
% % %P_plot2(P2,[9 15 19 16 17 18],P2_AnomalyIdx,PID); %P203

% PID = 3;
% P3 = P_preprocess(P3n, P3a, startTime,PID);
% P3 = P_preprocess_results(results_file_name,P3,anomalylikelihoodThreshold,PID);
% P3_AnomalyIdx = 13;
% P_plot2(P3,[1 18 13 17 15],P3_AnomalyIdx,PID); %FIT301


PID = 4;
P4 = P_preprocess(P4n, P4a, startTime,PID);
P4 = P_preprocess_results(results_file_name,P4,anomalylikelihoodThreshold,PID);
P4_AnomalyIdx = 14;
P_plot2(P4,[2 19 14 18 16],P4_AnomalyIdx,PID);


% PID = 5;
% P5 = P_preprocess(P5n, P5a, startTime,PID);
% P5 = P_preprocess_results(results_file_name,P5,anomalylikelihoodThreshold,PID);
% P5_AnomalyIdx = 14;
% P_plot2(P5,[6 19 14 18 16],P5_AnomalyIdx,PID); %FIT301


% PID = 6;
% P6 = P_preprocess(P6n, P6a, startTime,PID);
% P6 = P_preprocess_results(results_file_name,P6,anomalylikelihoodThreshold,PID);
% P6_AnomalyIdx = 6;
% P_plot2(P6,[1 11 6 10 8],P6_AnomalyIdx,PID); %FIT301
