using StatsPlots
using FileIO
using JLD2
using CodecZlib
using EM
using SpecialFunctions

include("../code/hmm.jl")
include("../code/util.jl")
include("../code/em_scripts.jl")

fns = [
       ("hmm_leaf", run_hmm_leaf),
       ("hmm_leaf_γ2", run_hmm_leaf_γ2),
       ("hmm_leaf_retainbelief", run_hmm_leaf_retainbelief),
       ("hmm_leaf_stay", run_hmm_leaf_stay),
       ("hmm_leaf_turn", run_hmm_leaf_turn),
       ("hmm_leaf_spatial", run_hmm_leaf_spatial),
       ("hmm_leaf_leafspatial", run_hmm_leaf_leafspatial),
       ("hmm_leaf_leafturn", run_hmm_leaf_leafturn),

       ("hmm_leaf_stay_γ2", run_hmm_leaf_stay_γ2),
       ("hmm_leaf_stay_retainbelief", run_hmm_leaf_stay_retainbelief),
       ("hmm_leaf_stay_turn", run_hmm_leaf_stay_turn),
       ("hmm_leaf_stay_spatial", run_hmm_leaf_stay_spatial),
       ("hmm_leaf_stay_leafturn", run_hmm_leaf_stay_leafturn),
       ("hmm_leaf_stay_leafspatial", run_hmm_leaf_stay_leafspatial),
       ("hmm_leaf_stay_turn_leafspatial", run_hmm_leaf_stay_turn_leafspatial),
       ("hmm_leaf_stay_turn_leafturn", run_hmm_leaf_stay_turn_leafturn),
       ("hmm_leaf_stay_spatial_leafspatial", run_hmm_leaf_stay_spatial_leafspatial),
       ("hmm_leaf_stay_spatial_leafturn", run_hmm_leaf_stay_spatial_leafturn),

       ("hmm_leaf_turn_γ2", run_hmm_leaf_turn_γ2),
       ("hmm_leaf_spatial_γ2", run_hmm_leaf_spatial_γ2),
       ("hmm_leaf_turn_leafspatial_γ2", run_hmm_leaf_turn_leafspatial_γ2),
       ("hmm_leaf_turn_leafturn_γ2", run_hmm_leaf_turn_leafturn_γ2),
       ("hmm_leaf_spatial_leafspatial_γ2", run_hmm_leaf_spatial_leafspatial_γ2),
       ("hmm_leaf_spatial_leafturn_γ2", run_hmm_leaf_spatial_leafturn_γ2),

       ("hmm_leaf_stay_turn_γ2", run_hmm_leaf_stay_turn_γ2),
       ("hmm_leaf_stay_spatial_γ2", run_hmm_leaf_stay_spatial_γ2),
       ("hmm_leaf_stay_leafturn_γ2", run_hmm_leaf_stay_leafturn_γ2),
       ("hmm_leaf_stay_leafspatial_γ2", run_hmm_leaf_stay_leafspatial_γ2),
       ("hmm_leaf_stay_turn_leafspatial_γ2", run_hmm_leaf_stay_turn_leafspatial_γ2),
       ("hmm_leaf_stay_turn_leafturn_γ2", run_hmm_leaf_stay_turn_leafturn_γ2),
       ("hmm_leaf_stay_spatial_leafspatial_γ2", run_hmm_leaf_stay_spatial_leafspatial_γ2),
       ("hmm_leaf_stay_spatial_leafturn_γ2", run_hmm_leaf_stay_spatial_leafturn_γ2),
]

function hmm_biases_dj(
    model_name, animal_name,
    trials_info_by_rat_csv_path,
    maxiter, extended, compress,
    behavior_model_fit_path,
    behavior_model_csv_path,
    RERUN, SAVE_Q,
    rewscaled
    )
    
    println("going to start running the hmm biases spyglass fxn for schema integration")
    
    #use animal name to get index in list animals
    animal_ind = findfirst(isequal(animal_name), animals)
    println(animal_ind)
    #use model name to get index in list fn
    fns_names = [fns[i][1] for i in range(1,(length(fns)))]
    fn_ind = findfirst(isequal(string("hmm_",model_name)), fns_names)
    
    animal = animals[animal_ind]
    (fname, fn) = fns[fn_ind]
    
    println("animal name is $(animal_name)")
    println(animal)
    println(fname)
    
    println("still going ok")
    
    println(fn)
    
    println("just printed fname then fn")
    
    println(pwd())
    
    #overriding my input parameters for now
    #trials_info_by_rat_csv_path = "../data/j16_clean_contingencies_only_parsed_data.csv"
    #behavior_model_fit_path = "../results/hmm_biases"
    #behavior_model_csv_path = "../results/hmm_Q_vals"
    
    data = load_animal_dj(animal, trials_info_by_rat_csv_path)
    
    println("loaded data with load animal dj fxn")
    
    if RERUN
        if rewscaled==false
            results = fn(data; maxiter=maxiter, extended=extended)
            fit_full_path = "$(behavior_model_fit_path)/$(fname)_$(animal).jld2"
            save(fit_full_path, "$(fname)_$(animal)", results; compress=compress)
        else
            results = fn(data; maxiter=maxiter, extended=extended, rewscaled=rewscaled)
            fit_full_path = "$(behavior_model_fit_path)/$(fname)_rewscaled_$(animal).jld2"
            save(fit_full_path, "$(fname)_rewscaled_$(animal)", results; compress=compress)
        end
    else
        results = load("$(behavior_model_fit_path)/hmm_$(fn)_$(animal).jld2", "hmm_$(fn)_$(animal)")
    end
    
    println("finished loading results and saving as jld2")
    
    Qs = find_Q_vals_by_day_dj(animal, results, trials_info_by_rat_csv_path, rewscaled)
    
    println("finished getting Qs with find q vals by day dj fxn")
    
    if SAVE_Q
        csv_full_path = "$(behavior_model_csv_path)/Q_vals_$(fn)_$(animal).csv"
        CSV.write(csv_full_path, Qs)
    end
    
    println("finished writing csv output yay")
    
    println("now i should return the output paths and potentially also the csv")
    
    return fit_full_path, csv_full_path, Qs
    
end

function find_Q_vals_by_day_dj(animal, results, trials_info_by_rat_csv_path, rewscaled; add_leaf=true)
    data = load_animal_dj(animal, trials_info_by_rat_csv_path)
    ndays = maximum(data.daynum)
    liks = zeros(ndays)
    dfs = []
    for i in 1:ndays
        (liks[i], df) = hmm_lik(view(data, data.daynum .== i, :), get_contingencies(), results;
            subject=i, add_leaf=add_leaf, rewscaled=rewscaled, record=true)
        push!(dfs, df)
    end
    # Combine all session results
    record_df = vcat(dfs...)
    # Append columns to the original data
    hcat(data, record_df)
end

"""
Return a dataframe for the animal

    animal<string>: Name of the animal
    depletion<bool, default false>: Whether to use the non-depletion or depletion data for this anmal

    Coding:
    reward: 0/1
    rewscaled: -1/1

    stem: A/B/C
    stemchoice: 1-3
    leaf: 1-6
    leafchoice: 1-2 (conditional on stem)

    trial: 0-179
    session: 1-?, per-day session number, e.g. 1 is first session each day
    daynum: 1-?, contiguous day numbering
    daysessionnum: 1-?, unique session number continued across days

"""
function load_animal_dj(animal, trials_info_by_rat_csv_path)
    # csv_file to estimate value for
    #if depletion
    #    csv_file = animal * "_clean_contingencies_only_parsed_depletion_data.csv"
    #    # mkdir(fullfile(filepath,["hmm_decay_",animal]))
    #else
    #    csv_file = animal * "_clean_contingencies_only_parsed_data.csv"
    #    # mkdir(fullfile(filepath,['hmm_',animal]))
    #end
    #fullpath = joinpath(filepath, "data", csv_file)
    #df = DataFrame(CSV.File(fullpath, drop=[1]))  # Drop index column
    df = DataFrame(CSV.File(trials_info_by_rat_csv_path, drop=[1]))  # Drop index column

    # recode some variables
    df.rewscaled = 2 * df.reward .- 1
    df.stemchoice = [df[i,:stem][1] - 'A' + 1 for i in 1:nrow(df)]
    df.leafchoice = 1 .+ mod.(df.leaf.+1,2)

    # Number days 1-n
    dates = unique(df.date)
    df.daynum = [minimum(findall(df[i,:date] .== dates)) for i in 1:nrow(df)]

    # Code sessions contiguously across days
    datesessions = string.(df.date) .* string.(df.session)
    uniq_datesessions = unique(datesessions)
    df.daysessionnum = [minimum(findall(datesessions[i] .== uniq_datesessions)) for i in 1:nrow(df)]

    # Stem switches
    df[!, :stemswitch] .= false
    prevstem = 0
    prevsession = 0
    for i in 1:size(df, 1)
        session = df[i, :session]
        if (session == prevsession)
            stem = df[i, :stem]
            if (stem != prevstem)
                df[i, :stemswitch] = true
            end
            prevstem = stem
        end
        prevsession = session
    end

    df.cont_combo = cont_combination_old.(string.(df.contingency))
    df.contingency = string.(df.contingency)

    return df
end