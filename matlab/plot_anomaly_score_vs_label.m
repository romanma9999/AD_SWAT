function plot_anomaly_score_vs_label(anomaly_score_filename,label_filename)


filepath = ['../HTM_results/' anomaly_score_filename];
Array=csvread(filepath,0,0);
anomaly_score = Array(:, 1);

filepath = ['../HTM_results/' label_filename];
Array=csvread(filepath,0,0);
label = Array(:, 1);

% fid  = fopen( filepath, 'r' ) ;
% anomaly_score = textscan( fid, '%d') ;
% fclose( fid ) ;
% 
% filepath = ['../HTM_results/' label_filename];
% fid  = fopen( filepath, 'r' ) ;
% label = textscan( fid, '%d') ;
% fclose( fid ) ;



figure;

x = 1:size(label,1);
plot(x,anomaly_score)
hold on
plot(x,2+label,'r')
grid on
ylabel('score');

