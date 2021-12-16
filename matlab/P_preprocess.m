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


[m,n] = size(P.Variables);
val = zeros(m,1);

switch PID
    case 1
         for i = 1:m
             val(i,1) = 1 + 4*P.MV101(i) + 2*(P.P101(i)-1) + (P.P102(i)-1);  %transform MV101(0/1/2),P101(1/2),P102(1/2) to one discrete variable with range 0-12, 0 is reserved for unknown 
         end
    case 2

end

P = addvars(P,val,'Before','Anomaly','NewVariableNames','CombinedState');

end




