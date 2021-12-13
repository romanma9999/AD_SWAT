function P =  P_preprocess_results(P,anomalylikelihoodThreshold,PID)

data = load_htm_results_data(['../HTM_results/P' num2str(PID) '_res.csv']);
data.title = "";

tmp = data.anomalylikelihood>anomalylikelihoodThreshold;
anomaly_tmp = data.anomalylikelihood;
anomaly_tmp(~tmp) = 0;
    
P = addvars(P,data.pred1,data.pred5,anomaly_tmp,data.anomaly,'NewVariableNames',{'Pred1','Pred5','AnomalyLikelihood','AnomalyScore'});
end

