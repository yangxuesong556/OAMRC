function Pop=quzheng(pop)

n=size(pop,1);
h=0.05:0.05:5;

for i=1:n
    Pop(i,1)=h(min(find(min(abs(h-pop(i,1)))==abs(h-pop(i,1)))));
end %h
Pop(:,2)=roundn(pop(:,2),-1);%g
Pop(:,3)=roundn(pop(:,3),-1); %a
Pop(:,4)=round(pop(:,4));%p
Pop(:,5)=round(pop(:,5));%delay_time
Pop(:,6)=round(pop(:,6));%layer
end
