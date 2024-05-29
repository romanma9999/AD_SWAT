clc
close all
clear all

param_res = 1;

subsampling = 1; repeat_training = 1; anomalylikelihoodThreshold = 0.6;


load('swat_nominal.mat');
[P1n,P2n,P3n,P4n,P5n,P6n] = parse_swat(swat_nominal,subsampling);
%  plot_swat(P1n,P2n,P3n,P4n,P5n,P6n);

load('swat_attack.mat');
[P1a,P2a,P3a,P4a,P5a,P6a] = parse_swat(swat_attack,subsampling);
%plot_swat(P1a,P2a,P3a,P4a,P5a,P6a);
startTime = '12/23/2015 15:00:00';
finishTrainingTime = datetime('28/12/2015 9:59:59','InputFormat','dd/MM/uuuu HH:mm:ss');


channel = 'P3_LIT301';
short_channel = 'LIT301';
PID = 3;
P = P_preprocess(P3n, P3a, startTime,PID);

file_names{1} = ['../hierarchy/diff/h1/HTM_results/' channel '_learn_always_freeze_off_res.csv'];
file_names{2} = ['../hierarchy/diff/h2/HTM_results/' channel '_learn_always_freeze_off_res.csv'];
file_names{3} = ['../hierarchy/diff/h3/HTM_results/' channel '_learn_always_freeze_off_res.csv'];
% file_names{4} = ['../hierarchy/abs/h5/HTM_results/' channel '_learn_always_freeze_off_res.csv'];

% file_names{1} = ['../hierarchy/abs/h1/HTM_results/' channel '_learn_train_only_freeze_off_res.csv'];
% file_names{2} = ['../hierarchy/abs/h2/HTM_results/' channel '_learn_train_only_freeze_off_res.csv'];
% file_names{3} = ['../hierarchy/abs/h3/HTM_results/' channel '_learn_train_only_freeze_off_res.csv'];
%file_names{4} = ['../hierarchy/abs/h5/HTM_results/' channel '_learn_train_only_freeze_off_res.csv'];

title_names{1} = [short_channel ' H1'];
title_names{2} = [short_channel ' H2'];
title_names{3} = [short_channel ' H3'];
%title_names{4} = [short_channel ' H5'];


n = length(file_names);
for i = 1:n
    tmp = load_htm_results_data(file_names{i});
    tmp.title = title_names{i}
    tmp.plot_type = 2;
    data(i) = tmp;    
end

plot_multiple_htm_results(data,P);

