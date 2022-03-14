using Plots
using StatsPlots
using FileIO
include("hmm.jl")
using EM

function hmm_lik_stay_turn_n1_fn(params, data)
    # these are the free parameters
    βgo = params[1]   # beta for switch to alternative stem 
    βstay = params[2] # beta for current stem
    βleaf = params[3] # beta for leaf choice on switch
    stay_bias = params[4]
    turn_bias = params[5]
    spatial_bias = [0.0, 0.0, 0.0]
    volatility = 0.1 + 0.1 * erf(params[6] / sqrt(2)) # volatility (squashed to 0-0.2 using standard normal CDF)
    ϕ = get_contingencies(n=1)
    return hmm_lik(data, βgo, βstay, βleaf, stay_bias, turn_bias, spatial_bias, volatility, ϕ)
end

function hmm_lik_stay_turn_n2_fn(params, data)
    # these are the free parameters
    βgo = params[1]   # beta for switch to alternative stem 
    βstay = params[2] # beta for current stem
    βleaf = params[3] # beta for leaf choice on switch
    stay_bias = params[4]
    turn_bias = params[5]
    spatial_bias = [0.0, 0.0, 0.0]
    volatility = 0.1 + 0.1 * erf(params[6] / sqrt(2)) # volatility (squashed to 0-0.2 using standard normal CDF)
    ϕ = get_contingencies(n=2)
    return hmm_lik(data, βgo, βstay, βleaf, stay_bias, turn_bias, spatial_bias, volatility, ϕ)
end

function hmm_lik_stay_turn_n3_fn(params, data)
    # these are the free parameters
    βgo = params[1]   # beta for switch to alternative stem 
    βstay = params[2] # beta for current stem
    βleaf = params[3] # beta for leaf choice on switch
    stay_bias = params[4]
    turn_bias = params[5]
    spatial_bias = [0.0, 0.0, 0.0]
    volatility = 0.1 + 0.1 * erf(params[6] / sqrt(2)) # volatility (squashed to 0-0.2 using standard normal CDF)
    ϕ = get_contingencies(n=3)
    return hmm_lik(data, βgo, βstay, βleaf, stay_bias, turn_bias, spatial_bias, volatility, ϕ)
end

function hmm_lik_stay_turn_n4_fn(params, data)
    # these are the free parameters
    βgo = params[1]   # beta for switch to alternative stem 
    βstay = params[2] # beta for current stem
    βleaf = params[3] # beta for leaf choice on switch
    stay_bias = params[4]
    turn_bias = params[5]
    spatial_bias = [0.0, 0.0, 0.0]
    volatility = 0.1 + 0.1 * erf(params[6] / sqrt(2)) # volatility (squashed to 0-0.2 using standard normal CDF)
    ϕ = get_contingencies(n=4)
    return hmm_lik(data, βgo, βstay, βleaf, stay_bias, turn_bias, spatial_bias, volatility, ϕ)
end

function hmm_lik_stay_turn_n5_fn(params, data)
    # these are the free parameters
    βgo = params[1]   # beta for switch to alternative stem 
    βstay = params[2] # beta for current stem
    βleaf = params[3] # beta for leaf choice on switch
    stay_bias = params[4]
    turn_bias = params[5]
    spatial_bias = [0.0, 0.0, 0.0]
    volatility = 0.1 + 0.1 * erf(params[6] / sqrt(2)) # volatility (squashed to 0-0.2 using standard normal CDF)
    ϕ = get_contingencies(n=5)
    return hmm_lik(data, βgo, βstay, βleaf, stay_bias, turn_bias, spatial_bias, volatility, ϕ)
end
function hmm_lik_stay_turn_n6_fn(params, data)
    # these are the free parameters
    βgo = params[1]   # beta for switch to alternative stem 
    βstay = params[2] # beta for current stem
    βleaf = params[3] # beta for leaf choice on switch
    stay_bias = params[4]
    turn_bias = params[5]
    spatial_bias = [0.0, 0.0, 0.0]
    volatility = 0.1 + 0.1 * erf(params[6] / sqrt(2)) # volatility (squashed to 0-0.2 using standard normal CDF)
    ϕ = get_contingencies(n=6)
    return hmm_lik(data, βgo, βstay, βleaf, stay_bias, turn_bias, spatial_bias, volatility, ϕ)
end
function hmm_lik_stay_turn_n7_fn(params, data)
    # these are the free parameters
    βgo = params[1]   # beta for switch to alternative stem 
    βstay = params[2] # beta for current stem
    βleaf = params[3] # beta for leaf choice on switch
    stay_bias = params[4]
    turn_bias = params[5]
    spatial_bias = [0.0, 0.0, 0.0]
    volatility = 0.1 + 0.1 * erf(params[6] / sqrt(2)) # volatility (squashed to 0-0.2 using standard normal CDF)
    ϕ = get_contingencies(n=7)
    return hmm_lik(data, βgo, βstay, βleaf, stay_bias, turn_bias, spatial_bias, volatility, ϕ)
end
function hmm_lik_stay_turn_n8_fn(params, data)
    # these are the free parameters
    βgo = params[1]   # beta for switch to alternative stem 
    βstay = params[2] # beta for current stem
    βleaf = params[3] # beta for leaf choice on switch
    stay_bias = params[4]
    turn_bias = params[5]
    spatial_bias = [0.0, 0.0, 0.0]
    volatility = 0.1 + 0.1 * erf(params[6] / sqrt(2)) # volatility (squashed to 0-0.2 using standard normal CDF)
    ϕ = get_contingencies(n=8)
    return hmm_lik(data, βgo, βstay, βleaf, stay_bias, turn_bias, spatial_bias, volatility, ϕ)
end

function run_hmm_em_stay_turn_n1(animal; maxiter=100)
    data = load_animal(animal)
    full = true
    fn = hmm_lik_stay_turn_n1_fn
    data[!, :sub] = string.(data.date)
    subs = unique(data[:,:sub]) #in this case subs is just differentiating days rather than rats/subjects
    NS = length(subs) #number of subjects/days
    X = ones(NS) # (group level design matrix); #x group level design matrix...
    # matrix across all days of this task)

    # fit the full model
    initbetas = [0. 0 0 0 0 0] 
    initsigma = [5., 5, 5, 5, 1, 1]
    varnames = ["βgo", "βstay", "βleaf", "stay_bias", "turn_bias", "volatility"]

    (betas,sigma,x,l,h,opt_rec) = em(data,subs,X,initbetas,initsigma,fn; emtol=1e-3, full=full, maxiter=maxiter);
    (standarderrors,pvalues,covmtx) = emerrors(data,subs,x,X,h,betas,sigma,fn)
    EMResultsExtended(varnames,betas,sigma,x,l,h,opt_rec,standarderrors,pvalues,covmtx)
end
function run_hmm_em_stay_turn_n2(animal; maxiter=100)
    data = load_animal(animal)
    full = true
    fn = hmm_lik_stay_turn_n2_fn
    data[!, :sub] = string.(data.date)
    subs = unique(data[:,:sub]) #in this case subs is just differentiating days rather than rats/subjects
    NS = length(subs) #number of subjects/days
    X = ones(NS) # (group level design matrix); #x group level design matrix...
    # matrix across all days of this task)

    # fit the full model
    initbetas = [0. 0 0 0 0 0] 
    initsigma = [5., 5, 5, 5, 1, 1]
    varnames = ["βgo", "βstay", "βleaf", "stay_bias", "turn_bias", "volatility"]

    (betas,sigma,x,l,h,opt_rec) = em(data,subs,X,initbetas,initsigma,fn; emtol=1e-3, full=full, maxiter=maxiter);
    (standarderrors,pvalues,covmtx) = emerrors(data,subs,x,X,h,betas,sigma,fn)
    EMResultsExtended(varnames,betas,sigma,x,l,h,opt_rec,standarderrors,pvalues,covmtx)
end
function run_hmm_em_stay_turn_n3(animal; maxiter=100)
    data = load_animal(animal)
    full = true
    fn = hmm_lik_stay_turn_n3_fn
    data[!, :sub] = string.(data.date)
    subs = unique(data[:,:sub]) #in this case subs is just differentiating days rather than rats/subjects
    NS = length(subs) #number of subjects/days
    X = ones(NS) # (group level design matrix); #x group level design matrix...
    # matrix across all days of this task)

    # fit the full model
    initbetas = [0. 0 0 0 0 0] 
    initsigma = [5., 5, 5, 5, 1, 1]
    varnames = ["βgo", "βstay", "βleaf", "stay_bias", "turn_bias", "volatility"]

    (betas,sigma,x,l,h,opt_rec) = em(data,subs,X,initbetas,initsigma,fn; emtol=1e-3, full=full, maxiter=maxiter);
    (standarderrors,pvalues,covmtx) = emerrors(data,subs,x,X,h,betas,sigma,fn)
    EMResultsExtended(varnames,betas,sigma,x,l,h,opt_rec,standarderrors,pvalues,covmtx)
end
function run_hmm_em_stay_turn_n4(animal; maxiter=100)
    data = load_animal(animal)
    full = true
    fn = hmm_lik_stay_turn_n4_fn
    data[!, :sub] = string.(data.date)
    subs = unique(data[:,:sub]) #in this case subs is just differentiating days rather than rats/subjects
    NS = length(subs) #number of subjects/days
    X = ones(NS) # (group level design matrix); #x group level design matrix...
    # matrix across all days of this task)

    # fit the full model
    initbetas = [0. 0 0 0 0 0] 
    initsigma = [5., 5, 5, 5, 1, 1]
    varnames = ["βgo", "βstay", "βleaf", "stay_bias", "turn_bias", "volatility"]

    (betas,sigma,x,l,h,opt_rec) = em(data,subs,X,initbetas,initsigma,fn; emtol=1e-3, full=full, maxiter=maxiter);
    (standarderrors,pvalues,covmtx) = emerrors(data,subs,x,X,h,betas,sigma,fn)
    EMResultsExtended(varnames,betas,sigma,x,l,h,opt_rec,standarderrors,pvalues,covmtx)
end
function run_hmm_em_stay_turn_n5(animal; maxiter=100)
    data = load_animal(animal)
    full = true
    fn = hmm_lik_stay_turn_n5_fn
    data[!, :sub] = string.(data.date)
    subs = unique(data[:,:sub]) #in this case subs is just differentiating days rather than rats/subjects
    NS = length(subs) #number of subjects/days
    X = ones(NS) # (group level design matrix); #x group level design matrix...
    # matrix across all days of this task)

    # fit the full model
    initbetas = [0. 0 0 0 0 0] 
    initsigma = [5., 5, 5, 5, 1, 1]
    varnames = ["βgo", "βstay", "βleaf", "stay_bias", "turn_bias", "volatility"]

    (betas,sigma,x,l,h,opt_rec) = em(data,subs,X,initbetas,initsigma,fn; emtol=1e-3, full=full, maxiter=maxiter);
    (standarderrors,pvalues,covmtx) = emerrors(data,subs,x,X,h,betas,sigma,fn)
    EMResultsExtended(varnames,betas,sigma,x,l,h,opt_rec,standarderrors,pvalues,covmtx)
end
function run_hmm_em_stay_turn_n6(animal; maxiter=100)
    data = load_animal(animal)
    full = true
    fn = hmm_lik_stay_turn_n6_fn
    data[!, :sub] = string.(data.date)
    subs = unique(data[:,:sub]) #in this case subs is just differentiating days rather than rats/subjects
    NS = length(subs) #number of subjects/days
    X = ones(NS) # (group level design matrix); #x group level design matrix...
    # matrix across all days of this task)

    # fit the full model
    initbetas = [0. 0 0 0 0 0] 
    initsigma = [5., 5, 5, 5, 1, 1]
    varnames = ["βgo", "βstay", "βleaf", "stay_bias", "turn_bias", "volatility"]

    (betas,sigma,x,l,h,opt_rec) = em(data,subs,X,initbetas,initsigma,fn; emtol=1e-3, full=full, maxiter=maxiter);
    (standarderrors,pvalues,covmtx) = emerrors(data,subs,x,X,h,betas,sigma,fn)
    EMResultsExtended(varnames,betas,sigma,x,l,h,opt_rec,standarderrors,pvalues,covmtx)
end
function run_hmm_em_stay_turn_n7(animal; maxiter=100)
    data = load_animal(animal)
    full = true
    fn = hmm_lik_stay_turn_n7_fn
    data[!, :sub] = string.(data.date)
    subs = unique(data[:,:sub]) #in this case subs is just differentiating days rather than rats/subjects
    NS = length(subs) #number of subjects/days
    X = ones(NS) # (group level design matrix); #x group level design matrix...
    # matrix across all days of this task)

    # fit the full model
    initbetas = [0. 0 0 0 0 0] 
    initsigma = [5., 5, 5, 5, 1, 1]
    varnames = ["βgo", "βstay", "βleaf", "stay_bias", "turn_bias", "volatility"]

    (betas,sigma,x,l,h,opt_rec) = em(data,subs,X,initbetas,initsigma,fn; emtol=1e-3, full=full, maxiter=maxiter);
    (standarderrors,pvalues,covmtx) = emerrors(data,subs,x,X,h,betas,sigma,fn)
    EMResultsExtended(varnames,betas,sigma,x,l,h,opt_rec,standarderrors,pvalues,covmtx)
end
function run_hmm_em_stay_turn_n8(animal; maxiter=100)
    data = load_animal(animal)
    full = true
    fn = hmm_lik_stay_turn_n8_fn
    data[!, :sub] = string.(data.date)
    subs = unique(data[:,:sub]) #in this case subs is just differentiating days rather than rats/subjects
    NS = length(subs) #number of subjects/days
    X = ones(NS) # (group level design matrix); #x group level design matrix...
    # matrix across all days of this task)

    # fit the full model
    initbetas = [0. 0 0 0 0 0] 
    initsigma = [5., 5, 5, 5, 1, 1]
    varnames = ["βgo", "βstay", "βleaf", "stay_bias", "turn_bias", "volatility"]

    (betas,sigma,x,l,h,opt_rec) = em(data,subs,X,initbetas,initsigma,fn; emtol=1e-3, full=full, maxiter=maxiter);
    (standarderrors,pvalues,covmtx) = emerrors(data,subs,x,X,h,betas,sigma,fn)
    EMResultsExtended(varnames,betas,sigma,x,l,h,opt_rec,standarderrors,pvalues,covmtx)
end

function senor()
    hmm_em_stay_turn_n1_senor = run_hmm_em_stay_turn_n1("senor")
    save("results/hmm_em_stay_turn_n1_senor.jld2", "hmm_em_stay_turn_n1_senor", hmm_em_stay_turn_n1_senor)
    hmm_em_stay_turn_n2_senor = run_hmm_em_stay_turn_n2("senor")
    save("results/hmm_em_stay_turn_n2_senor.jld2", "hmm_em_stay_turn_n2_senor", hmm_em_stay_turn_n2_senor)
    hmm_em_stay_turn_n3_senor = run_hmm_em_stay_turn_n3("senor")
    save("results/hmm_em_stay_turn_n3_senor.jld2", "hmm_em_stay_turn_n3_senor", hmm_em_stay_turn_n3_senor)
    hmm_em_stay_turn_n4_senor = run_hmm_em_stay_turn_n4("senor")
    save("results/hmm_em_stay_turn_n4_senor.jld2", "hmm_em_stay_turn_n4_senor", hmm_em_stay_turn_n4_senor)
    hmm_em_stay_turn_n5_senor = run_hmm_em_stay_turn_n5("senor")
    save("results/hmm_em_stay_turn_n5_senor.jld2", "hmm_em_stay_turn_n5_senor", hmm_em_stay_turn_n5_senor)
    hmm_em_stay_turn_n6_senor = run_hmm_em_stay_turn_n6("senor")
    save("results/hmm_em_stay_turn_n6_senor.jld2", "hmm_em_stay_turn_n6_senor", hmm_em_stay_turn_n6_senor)
    hmm_em_stay_turn_n7_senor = run_hmm_em_stay_turn_n7("senor")
    save("results/hmm_em_stay_turn_n7_senor.jld2", "hmm_em_stay_turn_n7_senor", hmm_em_stay_turn_n7_senor)
    hmm_em_stay_turn_n8_senor = run_hmm_em_stay_turn_n8("senor")
    save("results/hmm_em_stay_turn_n8_senor.jld2", "hmm_em_stay_turn_n8_senor", hmm_em_stay_turn_n8_senor)
end

function j16()
    hmm_em_stay_turn_n1_j16 = run_hmm_em_stay_turn_n1("j16")
    save("results/hmm_em_stay_turn_n1_j16.jld2", "hmm_em_stay_turn_n1_j16", hmm_em_stay_turn_n1_j16)
    hmm_em_stay_turn_n2_j16 = run_hmm_em_stay_turn_n2("j16")
    save("results/hmm_em_stay_turn_n2_j16.jld2", "hmm_em_stay_turn_n2_j16", hmm_em_stay_turn_n2_j16)
    hmm_em_stay_turn_n3_j16 = run_hmm_em_stay_turn_n3("j16")
    save("results/hmm_em_stay_turn_n3_j16.jld2", "hmm_em_stay_turn_n3_j16", hmm_em_stay_turn_n3_j16)
    hmm_em_stay_turn_n4_j16 = run_hmm_em_stay_turn_n4("j16")
    save("results/hmm_em_stay_turn_n4_j16.jld2", "hmm_em_stay_turn_n4_j16", hmm_em_stay_turn_n4_j16)
    hmm_em_stay_turn_n5_j16 = run_hmm_em_stay_turn_n5("j16")
    save("results/hmm_em_stay_turn_n5_j16.jld2", "hmm_em_stay_turn_n5_j16", hmm_em_stay_turn_n5_j16)
    hmm_em_stay_turn_n6_j16 = run_hmm_em_stay_turn_n6("j16")
    save("results/hmm_em_stay_turn_n6_j16.jld2", "hmm_em_stay_turn_n6_j16", hmm_em_stay_turn_n6_j16)
    hmm_em_stay_turn_n7_j16 = run_hmm_em_stay_turn_n7("j16")
    save("results/hmm_em_stay_turn_n7_j16.jld2", "hmm_em_stay_turn_n7_j16", hmm_em_stay_turn_n7_j16)
    hmm_em_stay_turn_n8_j16 = run_hmm_em_stay_turn_n8("j16")
    save("results/hmm_em_stay_turn_n8_j16.jld2", "hmm_em_stay_turn_n8_j16", hmm_em_stay_turn_n8_j16)
end

function wilbur()
    hmm_em_stay_turn_n1_wilbur = run_hmm_em_stay_turn_n1("wilbur")
    save("results/hmm_em_stay_turn_n1_wilbur.jld2", "hmm_em_stay_turn_n1_wilbur", hmm_em_stay_turn_n1_wilbur)
    hmm_em_stay_turn_n2_wilbur = run_hmm_em_stay_turn_n2("wilbur")
    save("results/hmm_em_stay_turn_n2_wilbur.jld2", "hmm_em_stay_turn_n2_wilbur", hmm_em_stay_turn_n2_wilbur)
    hmm_em_stay_turn_n3_wilbur = run_hmm_em_stay_turn_n3("wilbur")
    save("results/hmm_em_stay_turn_n3_wilbur.jld2", "hmm_em_stay_turn_n3_wilbur", hmm_em_stay_turn_n3_wilbur)
    hmm_em_stay_turn_n4_wilbur = run_hmm_em_stay_turn_n4("wilbur")
    save("results/hmm_em_stay_turn_n4_wilbur.jld2", "hmm_em_stay_turn_n4_wilbur", hmm_em_stay_turn_n4_wilbur)
    hmm_em_stay_turn_n5_wilbur = run_hmm_em_stay_turn_n5("wilbur")
    save("results/hmm_em_stay_turn_n5_wilbur.jld2", "hmm_em_stay_turn_n5_wilbur", hmm_em_stay_turn_n5_wilbur)
    hmm_em_stay_turn_n6_wilbur = run_hmm_em_stay_turn_n6("wilbur")
    save("results/hmm_em_stay_turn_n6_wilbur.jld2", "hmm_em_stay_turn_n6_wilbur", hmm_em_stay_turn_n6_wilbur)
    hmm_em_stay_turn_n7_wilbur = run_hmm_em_stay_turn_n7("wilbur")
    save("results/hmm_em_stay_turn_n7_wilbur.jld2", "hmm_em_stay_turn_n7_wilbur", hmm_em_stay_turn_n7_wilbur)
    hmm_em_stay_turn_n8_wilbur = run_hmm_em_stay_turn_n8("wilbur")
    save("results/hmm_em_stay_turn_n8_wilbur.jld2", "hmm_em_stay_turn_n8_wilbur", hmm_em_stay_turn_n8_wilbur)
end

function chimi()
    hmm_em_stay_turn_n1_chimi = run_hmm_em_stay_turn_n1("chimi")
    save("results/hmm_em_stay_turn_n1_chimi.jld2", "hmm_em_stay_turn_n1_chimi", hmm_em_stay_turn_n1_chimi)
    hmm_em_stay_turn_n2_chimi = run_hmm_em_stay_turn_n2("chimi")
    save("results/hmm_em_stay_turn_n2_chimi.jld2", "hmm_em_stay_turn_n2_chimi", hmm_em_stay_turn_n2_chimi)
    hmm_em_stay_turn_n3_chimi = run_hmm_em_stay_turn_n3("chimi")
    save("results/hmm_em_stay_turn_n3_chimi.jld2", "hmm_em_stay_turn_n3_chimi", hmm_em_stay_turn_n3_chimi)
    hmm_em_stay_turn_n4_chimi = run_hmm_em_stay_turn_n4("chimi")
    save("results/hmm_em_stay_turn_n4_chimi.jld2", "hmm_em_stay_turn_n4_chimi", hmm_em_stay_turn_n4_chimi)
    hmm_em_stay_turn_n5_chimi = run_hmm_em_stay_turn_n5("chimi")
    save("results/hmm_em_stay_turn_n5_chimi.jld2", "hmm_em_stay_turn_n5_chimi", hmm_em_stay_turn_n5_chimi)
    hmm_em_stay_turn_n6_chimi = run_hmm_em_stay_turn_n6("chimi")
    save("results/hmm_em_stay_turn_n6_chimi.jld2", "hmm_em_stay_turn_n6_chimi", hmm_em_stay_turn_n6_chimi)
    hmm_em_stay_turn_n7_chimi = run_hmm_em_stay_turn_n7("chimi")
    save("results/hmm_em_stay_turn_n7_chimi.jld2", "hmm_em_stay_turn_n7_chimi", hmm_em_stay_turn_n7_chimi)
    hmm_em_stay_turn_n8_chimi = run_hmm_em_stay_turn_n8("chimi")
    save("results/hmm_em_stay_turn_n8_chimi.jld2", "hmm_em_stay_turn_n8_chimi", hmm_em_stay_turn_n8_chimi)
end

function peanut()
    hmm_em_stay_turn_n1_peanut = run_hmm_em_stay_turn_n1("peanut")
    save("results/hmm_em_stay_turn_n1_peanut.jld2", "hmm_em_stay_turn_n1_peanut", hmm_em_stay_turn_n1_peanut)
    hmm_em_stay_turn_n2_peanut = run_hmm_em_stay_turn_n2("peanut")
    save("results/hmm_em_stay_turn_n2_peanut.jld2", "hmm_em_stay_turn_n2_peanut", hmm_em_stay_turn_n2_peanut)
    hmm_em_stay_turn_n3_peanut = run_hmm_em_stay_turn_n3("peanut")
    save("results/hmm_em_stay_turn_n3_peanut.jld2", "hmm_em_stay_turn_n3_peanut", hmm_em_stay_turn_n3_peanut)
    hmm_em_stay_turn_n4_peanut = run_hmm_em_stay_turn_n4("peanut")
    save("results/hmm_em_stay_turn_n4_peanut.jld2", "hmm_em_stay_turn_n4_peanut", hmm_em_stay_turn_n4_peanut)
    hmm_em_stay_turn_n5_peanut = run_hmm_em_stay_turn_n5("peanut")
    save("results/hmm_em_stay_turn_n5_peanut.jld2", "hmm_em_stay_turn_n5_peanut", hmm_em_stay_turn_n5_peanut)
    hmm_em_stay_turn_n6_peanut = run_hmm_em_stay_turn_n6("peanut")
    save("results/hmm_em_stay_turn_n6_peanut.jld2", "hmm_em_stay_turn_n6_peanut", hmm_em_stay_turn_n6_peanut)
    hmm_em_stay_turn_n7_peanut = run_hmm_em_stay_turn_n7("peanut")
    save("results/hmm_em_stay_turn_n7_peanut.jld2", "hmm_em_stay_turn_n7_peanut", hmm_em_stay_turn_n7_peanut)
    hmm_em_stay_turn_n8_peanut = run_hmm_em_stay_turn_n8("peanut")
    save("results/hmm_em_stay_turn_n8_peanut.jld2", "hmm_em_stay_turn_n8_peanut", hmm_em_stay_turn_n8_peanut)
end