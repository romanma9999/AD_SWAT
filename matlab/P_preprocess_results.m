function P =  P_preprocess_results(filename, P,anomalylikelihoodThreshold,PID)

data = load_htm_results_data(filename);
data.title = "";


anomaly_without_peaks = data.anomaly;
%tmp = anomaly_without_peaks>0.8;
%anomaly_without_peaks(tmp) = 0;

anomaly_mean = movmean(anomaly_without_peaks,[7,0]);
tmp = anomaly_mean>anomalylikelihoodThreshold;
anomaly_mean(~tmp) = 0;

s = init_log_anomaly_score(1000);
log_likelihood_anomaly = zeros(size(data.anomaly));

for i = 1:size(data.anomaly,1)
[s, log_likelihood_anomaly(i)] = compute_log_anomaly_score(s,data.anomaly(i));
end

tmp = log_likelihood_anomaly>anomalylikelihoodThreshold;
anomaly_log = log_likelihood_anomaly;
anomaly_log(~tmp) = 0;
anomaly_log = data.anomalylikelihood;

%anomaly_sum = movsum(anomaly_log,[119,0]);
anomaly_sum = movsum(anomaly_without_peaks,[119,0]);
tmp = anomaly_sum>1;
anomaly_sum(tmp) = 1;
tmp = anomaly_sum>anomalylikelihoodThreshold;
%anomaly_sum(~tmp) = 0;


P = addvars(P,anomaly_mean,anomaly_sum,anomaly_log,data.anomaly,data.inputs,'NewVariableNames',{'Temporal Mean Score (10 sec)','Temporal Sum Score (2 min)','Scaled Log Likelihood Score','Instant Score','Inputs'});

%P = addvars(P,data.pred1,data.pred5,anomaly_tmp,data.anomaly,'NewVariableNames',{'Pred1','Pred5','AnomalyLikelihood','AnomalyScore'});
end



function self = init_log_anomaly_score(period)
    self.period   = period;
    self.alpha    = 1.0 - exp(-1.0 / self.period);
    self.mean     = 1.0;
    self.var      = 0.0003;
    self.std      = sqrt(self.var);
    self.prev     = 0.0;
    self.n_records = 0;
end

function [s, res] = compute_log_anomaly_score(self, anomaly)

    likelihood = get_likelihood(self,anomaly);
    self = add_record(self,anomaly);
    combined = 1 - (1 - likelihood) * (1 - self.prev);
    self.prev = likelihood;
    if self.n_records < self.period
       res = 0.0;
    else
      res = get_log_likelihood(combined);
    end
    s = self;
    end

    function s = add_record(self, anomaly)

    diff      = anomaly - self.mean;
    incr      = self.alpha * diff;
    self.mean = self.mean + incr;
    self.var  = (1.0 - self.alpha) * (self.var + diff * incr);
    self.mean = max(self.mean, 0.03);
    self.var = max(self.var, 0.0003);
        
    self.std = sqrt(self.var);
    self.n_records = self.n_records + 1;
    s = self;
    end
    
    function res = get_likelihood(self, anomaly)
        z = (anomaly - self.mean) / self.std;
       res = 1.0 - 0.5 * erfc(z/1.4142); 
    end

    function res = get_log_likelihood(likelihood)
    %     Math.log(1.0000000001 - likelihood) / Math.log(1.0 - 0.9999999999)
        res = log(1.0000000001 - likelihood) / -23.02585084720009;
    end





