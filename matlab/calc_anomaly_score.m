function [outputArg1,outputArg2] = calc_anomaly_score(label, raw_anomaly, threshold)


anomaly_above_threshold = zeros(size(raw_anomaly));
anomaly_above_threshold(raw_anomaly > threshold) = 1;
score = dot(label, anomaly_above_threshold);

    
outputArg1 = inputArg1;
outputArg2 = inputArg2;
end

