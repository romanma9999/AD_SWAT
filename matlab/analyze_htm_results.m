clc
close all
clear all

param_res = 1;

%param_name = 'cellsPerColumn';values = [9, 10, 11, 12, 13];
%param_name = 'minThreshold';values = [8, 9, 10];
%param_name = 'activationThreshold';values = [12, 14, 16, 18, 20];
%param_name = 'initialPerm';values = [210, 310, 410, 510];
%param_name = 'maxSynapsesPerSegment';values = [64, 70, 96, 128];
param_name = 'permanenceDec'; values = [70 80 90 100];
%param_name = 'permanenceInc'; values = [50 60 70 80 90 100 110];

n = numel(values);
for i = 1:n
    tmp = load_htm_results_data(['../HTM_results/data4_' param_name '_' num2str(values(i)) '.csv']);
    tmp.title = [param_name '=' num2str(values(i)/param_res)];
    data(i) = tmp;    
end

plot_multiple_htm_results(data);

