function [P1,P2,P3,P4,P5,P6] = parse_swat(data,sampling_interval)

if nargin < 2
    sampling_interval =   1
  end

P1 = data(1:sampling_interval:end,{'LIT101','MV101','P101','P102','Normal'});
%P1 = data(:,{'FIT101','LIT101','MV101','P101','P102','Normal'});
P2 = data(1:sampling_interval:end,{'AIT201','AIT202','AIT203','FIT201','MV201','P201','P202','P203','P204','P205','P206','Normal'});
P3 = data(1:sampling_interval:end,{'DPIT301','FIT301','LIT301','MV301','MV302','MV303','MV304','P301','P302','Normal'});
P4 = data(1:sampling_interval:end,{'AIT401','AIT402','FIT401','LIT401','P401','P402','P403','P404','UV401','Normal'});
P5 = data(1:sampling_interval:end,{'AIT501','AIT502','AIT503','AIT504','FIT501','FIT502','FIT503','FIT504','P501','P502','PIT501','PIT502','PIT503','Normal'});
P6 = data(1:sampling_interval:end,{'FIT601','P601','P602','P603','Normal'});



% deprecate features MADICS
% variance is 0 :  P202, P301, P401, P404, P502, P601
% K-S Statistic higher than 0.25 : AIT402, AIT201, AIT501, AIT502, AIT202, AIT504, FIT301, PIT502, PIT503, FIT503, PIT501, AIT203, AIT401, AIT503, FIT601
% Training and test empirical distributions do not match : P201
%
% AIT202, AIT504
% yes : AIT202, AIT203, AIT402, AIT501,AIT502, AIT503, AIT504, FIT301, 
% yes : PIT501, PIT502, FIT503, FIT601,PIT503
% no : AIT201, AIT401


% P1 = data(:,{'FIT101','LIT101','MV101','P101','P102','Normal'});
% P2 = data(:,{'FIT201','MV201','P203','P204','P205','P206','Normal'});
% P3 = data(:,{'DPIT301','LIT301','MV301','MV302','MV303','MV304','P302','Normal'});
% P4 = data(:,{'FIT401','LIT401','P402','P403','P404','UV401','Normal'});
% P5 = data(:,{'FIT501','FIT502','FIT504','P501','Normal'});
% P6 = data(:,{'P602','P603','Normal'});


end

