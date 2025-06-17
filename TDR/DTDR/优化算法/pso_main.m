clear;
tic
popmin=[0.05,0.9,0.9,0,1];
popmax=[1,1,1,0,1];  %��ʼ��Χ    
maxgen=50; %��������
sizepop = 20;  %��Ⱥ��ģ100
Vmax = [0.075,0.15,0.15,1.5,1.5];       %�����ƶ�����ٶ�5
Vmin = [-0.075,-0.15,-0.15,-1.5,-1.5];      %�����ƶ���С�ٶ�-5
Dim=length(popmax) ;    %����ά��

wmax=0.9;wmin=0.4;
W=wmax*ones(1,sizepop);
c1 =1.5;       %��֪Ȩ������1.5
c2 =1.5;       %���Ȩ������1.5
c3=0;

re=sizepop*0.6;%ʣ����Ⱥ��ģ
record=zeros(maxgen,Dim);

%��������ӳ��
IN=2*rand(1,Dim)-ones(1,Dim);
inpop=[IN;zeros(sizepop-1,Dim)];
for i=1:sizepop-1
    inpop(i+1,:)=4*inpop(i,:).^3-3*inpop(i,:);
end

%ӳ�䵽����������
pop=zeros(sizepop,Dim);
for i=1:sizepop
   pop(i,:)=popmin+(ones(1,Dim)+inpop(i,:)).*(popmax-popmin)/2;
    
end
%pop(:,end)=round(pop(:,end))
pop=quzheng(pop);

for i = 1:sizepop
 V(i,:) = (Vmax-Vmin).*rand(1,Dim)+Vmin;       %��ʼ���ٶ�
    % ������Ӧ��
    %[fitness(i),RMSE(i,:),NRMSE(i,:)] = RC_main(pop(i,:),20);   %������Ӧ��(�����趨���ǽ����ֵԽС=��Ӧ��Խ��)
end
fitness=TDR_main(pop);% ������Ӧ��

toc
pop
fitness0=fitness;%������һ����Ӧ��ֵ������

[bestfitness bestindex] = min(fitness); %bestindex:ȫ��������������
gbest = pop(bestindex,:);   %ȫ�����λ��
pbest = pop;    %�������
fitnesspbest = fitness;   %���������Ӧ��ֵ
fitnessgbest = bestfitness;   %ȫ�������Ӧ��ֵ
record(1,:)=gbest;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%% ����
a00=zeros(1,maxgen);
for i = 1:maxgen       %��������
    for j = 1:sizepop  %��������
        % �ٶȸ���W(j)
        V(j,:) = 0.5*V(j,:) + c1*rand(1,Dim).*(pbest(j,:) - pop(j,:)) + c2*rand(1,Dim).*(gbest - pop(j,:));
        %�ٶȱ߽紦��
        V(j,find(V(j,:)>Vmax)) = Vmax(find(V(j,:)>Vmax));
        V(j,find(V(j,:)<Vmin)) = Vmin(find(V(j,:)<Vmin));
        
        % ��Ⱥ����
        pop(j,:) = pop(j,:) + V(j,:);
        %λ�ñ߽紦��
        
        pop(j,find(pop(j,:)>popmax)) = popmax(find(pop(j,:)>popmax));
        pop(j,find(pop(j,:)<popmin)) = popmin(find(pop(j,:)<popmin));
    end
   pop=quzheng(pop);
   fitness=TDR_main(pop);% ������Ӧ��

%  %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
% %�����Ŷ����ԡ��жϹ�������
%     if max(sqrt((fitness-mean(fitness)*ones(1,sizepop)).^2))>1
%         f=max(sqrt((fitness-mean(fitness)*ones(1,sizepop)).^2));
%     else
%         f=1;
%     end
%     sigma2=0;
%     for i1=1:sizepop
%         sigma2=sigma2+((fitness(i1)-mean(fitness))/f)^2;
%     end
%         %sigma2=sum(((fitness-mean(fitness))./f).^2)
%     sigma2;
% %��������ӳ��%�����ֲ����ŵ�
%     if sigma2<1
%         a00(i)=i;
%         REIN=2*rand(1,Dim)-ones(1,Dim);
%         reinpop=[REIN;zeros(re-1,Dim)];
%         for i1=1:re-1
%             reinpop(i1+1,:)=4*reinpop(i1,:).^3-3*reinpop(i1,:);
%         end
% 
%         %ӳ�䵽����������
%         repop=popmin*ones(re,Dim)+(ones(re,Dim)+reinpop).*(popmax-popmin)/2;
%         %������Ӧ��ֵ
%         for i1 = 1:re
%             refitness(i1) = fun(repop(i1,:),index);   %������Ӧ��(�����趨���ǽ����ֵԽС=��Ӧ��Խ��)
%         end
% %�滻
%         [veal,num]=sort(fitness);
%         for i1=sizepop-re+1:sizepop
%                 pop(num(i1),:)=repop(i1-(sizepop-re),:);
%         end   
%     end
%    
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%����    
        for j = 1:sizepop
        % �������Ÿ���
            if fitness(j) < fitnesspbest(j)
                pbest(j,:) = pop(j,:);
                fitnesspbest(j) = fitness(j);
            end
            % Ⱥ�����Ÿ���
            if fitness(j) < fitnessgbest
               gbest = pop(j,:);
            fitnessgbest = fitness(j);
            end
        end


%����Ȩ�ص�����Ӧ�仯
%     K=(fitness-fitness0)./fitness0;%������Ա仯��
%     W=(ones(1,sizepop)+exp(-1.*K)).^(-1);%����Ȩ�ص�����ʽ
 
    fitness0=fitness;
    
    fprintf('��%d����:  \n         ���Ž�:h = %f ,g = %f, a = %f, De= %f, La = %f \n         ȫ�������Ӧ��ֵ:%f\n'...
     ,i,gbest(1),gbest(2),gbest(3),gbest(4),gbest(5),fitnessgbest);  %����������ȡ1/fitnessgbest����Ϊ�����趨��ֵԽС����Ӧ��Խ��
 F(i)=fitnessgbest;
 %pop
  record(i+1,:)=gbest;
  toc
end
 F1=F;
figure(2)
plot(1:length(F1),F1)
title('ԭʼ�㷨����Ӧ�ȱ仯����');
fitnessgbest
find(a00~=0)
