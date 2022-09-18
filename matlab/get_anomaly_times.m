function [A,PA_idx]  = get_anomaly_times(PID)

A(1).s = '12/28/2015 10:29:14'; %hard [1] SS, trivial min/max
A(1).e = '12/28/2015 10:44:53'; 
A(2).s = '12/28/2015 10:50:46'; % [1,2], SS, trivial min/max
A(2).e = '12/28/2015 10:58:30';
A(3).s = '12/28/2015 11:22:00'; % [1] SS, abnormal signal shape, min/max will probably find it also..
A(3).e = '12/28/2015 11:28:22';
%4 - no impact
%5
A(6).s = '12/28/2015 12:00:55';% [2] SS, trivial min/max
A(6).e = '12/28/2015 12:04:10';
A(7).s = '12/28/2015 12:08:25';% [3] SS, trivial min/max
A(7).e = '12/28/2015 12:15:33';
A(8).s = '12/28/2015 13:10:10';% [3,4,6] SS, 
A(8).e = '12/28/2015 13:26:13';
    % P6 : P602/FIT601 freq change, 
    % P4: Lit401 pattern change, eventually value is below steady state min
    % P3 : DPIT301 trivial min/max, others freq change
%9
A(10).s = '12/28/2015 14:16:20';% [4,5] SS, trivial min / max
A(10).e = '12/28/2015 14:18:49'; % UPDATED TIME TO SEPARATE ANOMALY LABELS, ORIGINALLY 19:00
A(11).s = '12/28/2015 14:19:11';% [4,5] SS, trivial min / max 
A(11).e = '12/28/2015 14:28:20';
%12
%13 - no impact
%14 - no impact
%15
A(16).s = '12/29/2015 11:57:25';% [3] SS,  trivial min/max
A(16).e = '12/29/2015 12:02:00';
A(17).s = '12/29/2015 14:38:12';% [3,4] SS
A(17).e = '12/29/2015 14:50:08';
    % P4: Lit401 pattern change, eventually value is slighlty below steady state min. relatively interesting anomaly
    % P3: M301 stuck and hold value, relatively interesting anomaly
%18
A(19).s = '12/29/2015 18:10:43';% [5] SS, trivial value fixed
A(19).e = '12/29/2015 18:15:01';
A(20).s = '12/29/2015 18:15:43';% [5] SS, trivial min/max
A(20).e = '12/29/2015 18:22:17';
A(21).s = '12/29/2015 18:29:59';% [1] SM, quick (step?) change. 
A(21).e = '12/29/2015 18:42:00';
A(22).s = '12/29/2015 22:55:18';% [4,5] MM, trivial min/max
A(22).e = '12/29/2015 23:03:00';
A(23).s = '12/30/2015 01:42:34';% [3,4,6] MM, 
    % P6 : P602/FIT601 freq change
    % P4: Lit401 pattern change, eventually value is below steady state min
    % P3 : trivial min/max
A(23).e =  '12/30/2015 01:54:10';
A(24).s = '12/30/2015 09:51:08';% [2] SM, trivial min/max but not on the attack target channels
A(24).e = '12/30/2015 09:56:28';
A(25).s = '12/30/2015 10:01:50';% [4] SM
    % P4: Lit401 pattern change suddenly,  pattern is altered
A(25).e ='12/30/2015 10:12:01';
A(26).s = '12/30/2015 17:04:31';% [1,2,3] MS
    % P3: value stays constant
    % P2: trivial min/max
    %P1: relatively interesting pattern change, min/max will discover also    
A(26).e = '12/30/2015 17:29:00';
A(27).s = '12/31/2015 01:17:08';% [3,4] MS
    % P4: Lit410 value suddenly below min. 
    % P3: values stay constant longer than usual 
A(27).e =  '12/31/2015 01:44:58';% UPDATED TIME TO SEPARATE ANOMALY LABELS, ORIGINALLY 45:18
A(28).s = '12/31/2015 01:45:19';% [1,2,3,4,5,6] SS , P6 : P602/FIT601 freq change, P1-P5 : values stay constant for long period of time
A(28).e =   '12/31/2015 11:15:27';
% 29  - no impact
A(30).s = '12/31/2015 15:47:27';% [1,2] MM, 
    %P2 :  trivial min/max
    %P1 : value suddenly freezes and then sudden drop discoverable trivially by min/max
A(30).e = '12/31/2015 16:07:10';
A(31).s = '12/31/2015 22:05:34';% [4] SS
    % P4: Lit410 value suddenly below min. pattern change 
    
A(31).e = '12/31/2015 22:11:40';
A(32).s = '01/1/2016 10:36:00';% [3] SS, trivial min/max
A(32).e ='01/1/2016 10:46:00';
A(33).s = '01/01/2016 14:21:12';% [1] SS: value suddenly freezes and then sudden drop discoverable trivially by min/max
A(33).e = '01/01/2016 14:28:35';
A(34).s = '01/01/2016 17:12:40';% [1] SS, trivial min/max
A(34).e = '01/01/2016 17:14:20';
A(35).s = '01/01/2016 17:18:56'; %hard % [1] SM, interesting anomaly !!!
A(35).e = '01/01/2016 17:26:56';
A(36).s = '01/01/2016 22:16:01';% [1] SS, trivial min/max
A(36).e = '01/01/2016 22:25:00';
A(37).s = '01/2/2016 11:17:02';% [4,5] SM, trivial min/max
A(37).e ='01/2/2016 11:24:50';
A(38).s = '01/2/2016 11:31:38';% [4,5] MS,trivial min/max
A(38).e ='01/2/2016 11:36:18';
A(39).s = '01/2/2016 11:43:48';% [4,5] MS, value fixed (P5), trivial min/max (P4)
A(39).e = '01/2/2016 11:50:28';
A(40).s = '01/2/2016 11:51:42';% [4,5] SS, trivial min/max
A(40).e ='01/2/2016 11:56:38';
A(41).s = '01/2/2016 13:13:02';% [3] SS, trivial min/max
A(41).e = '01/2/2016 13:40:56';
%PA_idx = [];
PA_idx = [1 2 3 6 7 8 10 11 16 17 19:28 30:41];
%removed 28 from all..
%  switch PID
%          case 1
%              PA_idx = [1 2 3                                            21                      26  27    28 30           33 34 35 36]; % not min/max : 3?, 21, 26 (eventually),28(long constant value), 30(value freeze then minmax), 33(value freeze then minmax), 35
%          case 2
%              %  FIT201(2,26,28,30), AIT202(6,28),P203(24,28)
%             PA_idx = [   2     6                                                        24      26  27     28 30]; % not min/max :28(long constant value)
%          case 3
%              %LIT301(7,16,26,26,32,41),DPIT301(8,23,27,28),MV301(17,28)
%             PA_idx = [             7 8            16 17                       23           26  27 28          32                                              41];  % not min/max : 17, 26 (constant value), 27, 28(long constant value)
%          case 4
%              %LIT401(8,17,23,25,27,28, 31) ,AIT402(10,11,22,28,38,40),FIT401(10,11,22,37,39,40)
%             PA_idx = [                8 10 11       17                 22 23      25       27  28     31                             37 38 39 40];  % not min/max :  8(eventually) , 17(eventually), 23(eventually),25, 27?, 28(long constant value)
%          case 5
%             PA_idx = [                   10 11            19 20       22                            27  28                                     37 38 39 40]; % not min/max : 19 (value fixed), 28(long constant value), 39 (value fixed)
%          case 6
%             PA_idx = [                8                                             23                        27 28 ];  % not min/max : 8,23 , 28(long constant value)
%  end

end

