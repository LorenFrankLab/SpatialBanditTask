using CSV
using DataFrames
using MAT

function load_behavior_data(file,hmm_prefix)
    
    df = CSV.read(file, DataFrame)

    # recode some variables
    df.rewscaled = 2 * df.reward .- 1 #get reward to -1 or 1 no or yes instead of 0 to 1
    df.stemchoice = [df[i,:stem][1] - 'A' + 1 for i in 1:nrow(df)] #get stem choice to be represented as 1 2 3 instead of A B C
    df.leafchoice = 1 .+ mod.(df.leaf.+1,2)  #leaf choice keep represented as 1-6 (12 are on stem 1, 34 are on stem 2, and 56 on stem 3)

    dates = unique(df.date)
    df.daynum = [minimum(findall(df[i,:date] .== dates)) for i in 1:nrow(df)] 
    
    # break up / index sessions by session
    # dates_session = unique(df[:,[:date,:session]])
    # df.sub = [minimum(findall(dropdims(sum(Int32.(Vector(df[i,[:date,:session]])'.==Matrix(dates_session)),dims=2),dims=2).==2)) for i in 1:nrow(df)] 
    
    # break up / index sessions by day
    df.sub = df.daynum
    
    if length(hmm_prefix)>0
        Q=zeros(6,size(df,1))
        for day=1:length(dates)

            ss_result = matread(string(hmm_prefix,day,"_q_value.mat"))
            Q_tmp=ss_result["Q"];
            
            Q[:,df.daynum.==day]=Q_tmp;
        end
        df.q1=Q[1,:];
        df.q2=Q[2,:];
        df.q3=Q[3,:];
        df.q4=Q[4,:];
        df.q5=Q[5,:];
        df.q6=Q[6,:];
    end

    return df
    
end