function  [data,action_num,lines_of_action]  = get_data( subset,subject_set )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
data=[];
lines_of_action=[];
action_num=0;
subset_num=size(subset,2);
subject_num=size(subject_set,2);

%according to the input args,read the corresponding text file
for a=1:subset_num
    for s=1:subject_num
        for e=1:3
            file=sprintf('MSRAction3DSkeleton/a%02i_s%02i_e%02i_skeleton.txt',subset(a),subject_set(s),e);
            fp=fopen(file);
            if(fp>0)
                A=fscanf(fp,'%f');
                %add the label to A
                l=size(A,1)/4;
                A=reshape(A,4,l);
                temp=subset(a)*ones(1,l);
                A=[temp;A];
                A=reshape(A,5*l,1);
                %store the data into return value
                action_num=action_num+1;
                lines_of_action(action_num)=l;
                data=[data;A];
                
                fclose(fp);
            end
            
            
        end
    end
end

data_length=size(data,1)/5;
data=reshape(data,5,data_length);
data=data';
