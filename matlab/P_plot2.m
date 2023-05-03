function  P_plot2(P,plot_cfg,AnomalyIdx,threshold,PID)

[PA,PA_idx] = get_anomaly_times(PID);
nPA = length(PA_idx);
Pnames =P.Properties.VariableNames;
   
Pvar = P.Variables;

if isempty(plot_cfg) 
     n_var_to_plot = size(Pvar,2);
     var_to_plot = 1:n_var_to_plot;
else    
    var_to_plot = plot_cfg(1,:);
    n_var_to_plot = length(var_to_plot);
end

Pvar = (Pvar - min(Pvar))./(max(Pvar) - min(Pvar));
Ptime = P.Time;

ScoreIdx = var_to_plot(n_var_to_plot);

figure;
for i = 1:n_var_to_plot
    idx = var_to_plot(i);
    addon = (n_var_to_plot - i)*2;
     v = Pvar(:,idx);
     if isnan(v)
         v = zeros(size(Pvar(:,idx)));
     end
    if idx == AnomalyIdx 
      plot(Ptime,addon + v,'r','LineWidth',1,'DisplayName','Attack');
      hold on;
    else
     
      plot(Ptime,addon + v,'DisplayName',Pnames{idx});
      hold on;
      if idx == ScoreIdx && threshold > 0
         v = ones(size(Pvar(:,idx)))*threshold;
        plot(Ptime,addon + v,':r','LineWidth',1,'DisplayName','detection threshold');
     end
    xlabel('datetime');
    grid on;
end
legend show

end