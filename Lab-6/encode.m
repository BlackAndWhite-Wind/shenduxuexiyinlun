function [encode_data,inter_data]=encode(data)
    encode_data=cell(1,5);
    inter_data=int32(data);
    inter_data(1:2)=inter_data(1:2)-64;
    inter_data(3:end)=inter_data(3:end)-48;
    x1=zeros(5,1);x1(inter_data(1))=1;
    x2=zeros(5,1);x2(inter_data(2))=1;
    encode_data{1}=[ x1 ;x2 ];
    for i = 2:5
       var=zeros(3,1);var(inter_data(i+1))=1;
       encode_data{i}=var;
    end
    
    
