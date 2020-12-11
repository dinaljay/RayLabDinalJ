

clear  all;
Data = readtable('CT_LOOCV_HC_Mild_Open.csv');
% Data = readtable('CT_LOOCV_HC_Mod_Open.csv');


%true positive
TP=length(find(count(Data{:,2},'X0') & count(Data{:,3},'1')))
TN=length(find(count(Data{:,2},'X1') & count(Data{:,3},'2')))
FP=length(find(count(Data{:,2},'X1') & count(Data{:,3},'1')))
FN=length(find(count(Data{:,2},'X0') & count(Data{:,3},'2')))
% Accuracy: 
Acc=[TP+TN]/[TP+TN+FP+FN]
%sensitivity ( true postive rate)
 TPR=TP/[TP+FN]
%specificity 
 TNR=TN/[TN+FP]


































% 
% % just testing: 
% 
% T = readtable('/net/zfs-nil03/NIL03/hawasli/GBM300/SL_LL_SigOnly_FCroiwise_for_SVM.csv');
% 
% % less p-value
% clear sig_col;
% sig_col=[1 2]; % alreasding keping subjectID and group:
% ind1=find(table2array(T(:,2))==1); % index of grpup=1;
% ind0=find(table2array(T(:,2))==0);% index of grpup=0;
% 
% for ii = 3:size(T,2)
%            % [h p]=ttest2(double(data(ind0,ii)),double(data(ind1,ii)));
%              p=ranksum((table2array(T(ind0,ii))),table2array(T(ind1,ii)));
%             if p<0.005 
%                  sig_col=[sig_col p];
%              end
% end

