function P = P_preprocess(Pn,Pa,startTime, PID)

    PT = [Pn; Pa];
    PT.Normal(:) = 0;

    TR = timerange(startTime,'inf');
    P =  PT(TR,:);

    [PA, PA_idx] = get_anomaly_times(PID);
    
    nPA = length(PA_idx);
    P.Properties.VariableNames{end} = 'Anomaly';

    for j = 1:nPA
      dstart = datetime(PA(PA_idx(j)).s,'InputFormat','MM/dd/uuuu HH:mm:ss');
      dend = datetime(PA(PA_idx(j)).e,'InputFormat','MM/dd/uuuu HH:mm:ss');
      TR = timerange(dstart,dend);
      P(TR,:).Anomaly(:) = 1;
    end
end