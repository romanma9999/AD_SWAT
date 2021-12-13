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

% subplot(3,1,1)
% title("Predictions")
% xlabel("Time")
% ylabel("Value")
% x = 1:length(data.inputs);
% plot(x, data.inputs, 'red',x,data.pred1, 'blue',x, data.pred5, 'green')
% legend('Input', '1 Step Prediction, Shifted 1 step', '5 Step Prediction, Shifted 5 steps')
% %plot(x, data.inputs, 'red',x,data.pred1, 'blue')
% %legend('Input', '1 Step Prediction, Shifted 1 step')
% grid on
% 
% norm_inputs = (data.inputs-min(data.inputs))./(max(data.inputs) - min(data.inputs));
% 
% subplot(3,1,2)
% plot(x,norm_inputs,'b');
% xlabel("Time")
% ylabel("Value")
% grid on
% hold on
% plot_data(x,data,anomalylikelihoodThreshold,2);
% 
% subplot(3,1,3)
% plot(x,norm_inputs,'b');
% xlabel("Time")
% ylabel("Value")
% grid on
% hold on
% plot_data(x,data,anomalylikelihoodThreshold,4);
% 
% 
% 
