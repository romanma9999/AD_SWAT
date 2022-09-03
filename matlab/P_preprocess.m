function P = P_preprocess(Pn,Pa,startTime, PID)

% unite training and attack parts of the dataset
 PT = [Pn; Pa];
 PT.Normal(:) = 0;

% crop beginning
    TR = timerange(startTime,'inf');
    P =  PT(TR,:);

% mark the anomalies relevant to PID
    [PA, PA_idx] = get_anomaly_times(PID);
    
    nPA = length(PA_idx);
    P.Properties.VariableNames{end} = 'Attack';

    for j = 1:nPA
      dstart = datetime(PA(PA_idx(j)).s,'InputFormat','MM/dd/uuuu HH:mm:ss');
      dend = datetime(PA(PA_idx(j)).e,'InputFormat','MM/dd/uuuu HH:mm:ss');
      TR = timerange(dstart,dend);
      P(TR,:).Attack(:) = 1;
    end


% add Combined State variable
[m,n] = size(P.Variables);
d1 = zeros(m,1);
d2 = zeros(m,1);
d3 = zeros(m,1);
d4 = zeros(m,1);
d5 = zeros(m,1);
d6 = zeros(m,1);
d7 = zeros(m,1);
d8 = zeros(m,1);

%wstd = 10;
%mstd_name = ['MSTD' num2str(wstd)];
switch PID
    case 1
         d1(2:end) = abs(diff(P.LIT101));
         
         P = addvars(P,d1,'Before','Attack','NewVariableNames','ADIFF_LIT101');

%         P = addvars(P,d2,'Before','Anomaly','NewVariableNames',[mstd_name '_LIT101']);
    case 2
        d2(2:end) = abs(diff(P.AIT202));
        
%         d = diff(P.FIT201);
%         maxx = max(d);
%         minn = min(d);
%         diff_norm = (d - minn)/(maxx - minn);
%         diff_norm = diff_norm - mean(diff_norm);
%         diff_ang = atan(diff_norm*100)*180/pi;
%        
%         %d3(2:end) = diff(P.FIT201));
%         figure;
%         plot(P.FIT201);
%         title('FIT201')
%         figure;
%         plot(diff_norm*100)
%         title('diff norm')
%         figure;
%         plot(diff_ang)
%         title('diff ang')
%         figure;
%         histogram(P.FIT201,'Normalization','probability')
%         title('histogram FIT201')
%         figure;
%         histogram(diff_ang,90,'Normalization','probability')
%         title('diff ang')

%        d4 = movstd(P.AIT201,[wstd,0]);
%       d5 = movstd(P.AIT202,[wstd,0]);
%        d6 = movstd(P.AIT203,[wstd,0]);

%         P = addvars(P,d1,'Before','Attack','NewVariableNames','ADIFF_AIT201');
%         P = addvars(P,d4,'Before','Anomaly','NewVariableNames',[mstd_name '_AIT201']);
         P = addvars(P,d2,'Before','Attack','NewVariableNames','ADIFF_AIT202');
%         P = addvars(P,d5,'Before','Anomaly','NewVariableNames',[mstd_name '_AIT202']);
%         P = addvars(P,d3,'Before','Attack','NewVariableNames','ADIFF_AIT203');
%         P = addvars(P,d6,'Before','Anomaly','NewVariableNames',[mstd_name '_AIT203']);
    case 3
        d1(2:end) = abs(diff(P.DPIT301));
        d2(2:end) = abs(diff(P.FIT301));
        d3(2:end) = abs(diff(P.LIT301));
%        d4 = movstd(P.DPIT301,[wstd,0]);
%        d5 = movstd(P.FIT301,[wstd,0]);
%        d6 = movstd(P.LIT301,[wstd,0]);
         
        P = addvars(P,d1,'Before','Attack','NewVariableNames','ADIFF_DPIT301');
%        P = addvars(P,d4,'Before','Anomaly','NewVariableNames',[mstd_name '_DPIT301']); 
        P = addvars(P,d2,'Before','Attack','NewVariableNames','ADIFF_FIT301');
%        P = addvars(P,d5,'Before','Anomaly','NewVariableNames',[mstd_name '_FIT301']);
         P = addvars(P,d3,'Before','Attack','NewVariableNames','ADIFF_LIT301');      
%         P = addvars(P,d6,'Before','Anomaly','NewVariableNames',[mstd_name '_LIT301']);
    case 4
        d1(2:end) = abs(diff(P.AIT401));
        d2(2:end) = abs(diff(P.AIT402));
        d3(2:end) = abs(diff(P.FIT401));
        d4(2:end) = abs(diff(P.LIT401));
%        d5 = movstd(P.AIT401,[wstd,0]);
%        d6 = movstd(P.AIT402,[wstd,0]);
%        d7 = movstd(P.FIT401,[wstd,0]);
%        d8 = movstd(P.LIT401,[wstd,0]);
        
         P = addvars(P,d1,'Before','Attack','NewVariableNames','ADIFF_AIT401');
%         P = addvars(P,d5,'Before','Anomaly','NewVariableNames',[mstd_name '_AIT401']);
         P = addvars(P,d2,'Before','Attack','NewVariableNames','ADIFF_AIT402');
%         P = addvars(P,d6,'Before','Anomaly','NewVariableNames',[mstd_name '_AIT402']);
         P = addvars(P,d3,'Before','Attack','NewVariableNames','ADIFF_FIT401');
%         P = addvars(P,d7,'Before','Anomaly','NewVariableNames',[mstd_name '_FIT401']);
         P = addvars(P,d4,'Before','Attack','NewVariableNames','ADIFF_LIT401');
%         P = addvars(P,d8,'Before','Anomaly','NewVariableNames',[mstd_name '_LIT401']);
    case 5
        d1(2:end) = abs(diff(P.AIT501));
        d2(2:end) = abs(diff(P.AIT502));
        d3(2:end) = abs(diff(P.AIT503));
        d4(2:end) = abs(diff(P.AIT504));
        d5(2:end) = abs(diff(P.FIT501));
        d6(2:end) = abs(diff(P.FIT502));
        d7(2:end) = abs(diff(P.FIT503));
        d8(2:end) = abs(diff(P.FIT504));
%          P = addvars(P,d1,'Before','Anomaly','NewVariableNames','ADIFF_AIT501');
%          P = addvars(P,d2,'Before','Anomaly','NewVariableNames','ADIFF_AIT502');
%          P = addvars(P,d3,'Before','Anomaly','NewVariableNames','ADIFF_AIT503');        
%          P = addvars(P,d4,'Before','Anomaly','NewVariableNames','ADIFF_AIT504');        
%          P = addvars(P,d5,'Before','Anomaly','NewVariableNames','ADIFF_FIT501');
%          P = addvars(P,d6,'Before','Anomaly','NewVariableNames','ADIFF_FIT502');
%          P = addvars(P,d7,'Before','Anomaly','NewVariableNames','ADIFF_FIT503');        
%          P = addvars(P,d8,'Before','Anomaly','NewVariableNames','ADIFF_FIT504');        
     case 6
         d1(2:end) = abs(diff(P.FIT601));
%         d2 = movstd(P.FIT601,[wstd,0]);
         P = addvars(P,d1,'Before','Attack','NewVariableNames','ADIFF_FIT601');
%         P = addvars(P,d2,'Before','Anomaly','NewVariableNames',[mstd_name '_FIT601']);
end



end




