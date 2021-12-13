function plot_data(x,data,anomalylikelihoodThreshold,plot_type)
    tmp = data.anomalylikelihood>anomalylikelihoodThreshold;
    anomaly_tmp = data.anomalylikelihood;
    anomaly_tmp(~tmp) = 0;

    switch plot_type
         case 1
              sliding_window_anomaly = movmean(data.anomaly,[3 0]);
              plot(x,sliding_window_anomaly,'red','LineWidth',2)
              legend('anomaly score')
              title(['average anomaly score' data.title]);
         case 2
              plot(x,data.anomaly,'red','LineWidth',2)
              legend('anomaly score')
              title(['anomaly score' data.title]);
         case 3
              sliding_window_anomaly = movmean(data.anomalylikelihood,[3 0]);
              plot(x,sliding_window_anomaly,'red','LineWidth',2)
              legend('anomaly likelihood')
              title(['average anomaly likelihood  ' data.title]);
         case 4
               plot(x,anomaly_tmp,'red','LineWidth',2)
               legend('Input','anomaly likelihood')
               title(['anomaly likelihood > ' num2str(anomalylikelihoodThreshold) '  ' data.title]);
     end
   
    xlabel("Time")
    ylabel("Value")
    grid on
end
