function  p_prepare_for_HTM(Pn,Pa,startTime, finishTrainingTime,PID)
stage_name = ['P' num2str(PID)];
disp(['preprocess ' stage_name]);
% unite training and attack parts  
% crop beginning
% add 'Anomalies' variable with anomalies relevant to PID
P = P_preprocess(Pn, Pa, startTime,PID);
%P_plot(P1,[1 2 3 4 5 6],PID);
%P_plot(P,[],PID);

TrainSamplesCount = find(P.Time == finishTrainingTime);
disp(['train samples count is ' num2str(TrainSamplesCount)]);
disp(['save HTM input data for ' stage_name]);
sensors_data = P.Variables;
n_all_data = size(sensors_data,2); %remove 'Anomaly' label

%n_all_data = size(sensors_data,2)-1; %remove 'Anomaly' label
%sensors_data = sensors_data(:,1:n_all_data);

writematrix(sensors_data,['../HTM_input/' stage_name '_data.csv']) 

meta_data{1} = stage_name;
meta_data{2} = TrainSamplesCount;
meta_data{3} = n_all_data;
for i  = 1:n_all_data
 meta_data{1 + 3*i} = P.Properties.VariableNames{i}; 
 meta_data{2 +3*i} = floor(min(sensors_data(:,i)));
 meta_data{3 + 3*i} = ceil(max(sensors_data(:,i)));
end
writecell(meta_data,['../HTM_input/' stage_name '_meta.csv']) 

end

