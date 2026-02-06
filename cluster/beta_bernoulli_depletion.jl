using Logging
using JLD2
using EM
include("../code/beta_bernoulli.jl")
include("../code/util.jl")
include("../code/em_scripts.jl")


run_beta_lik_leaf_stay_turn_decay_γ2_depletion(data; kwargs...) = run_beta_lik(
    data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_beta_decay=true, add_γ2=true, add_depletion_factor=true, kwargs...)
run_beta_lik_leaf_stay_turn_decay_γ2(data; kwargs...) = run_beta_lik(
    data; add_βleaf=true, add_stay_bias=true, add_turn_bias=true, add_beta_decay=true, add_γ2=true, kwargs...)

# run_beta_lik_stay_turn_decay_γ2_depletion(data; kwargs...) = run_beta_lik(
#     data; add_βleaf=false, add_stay_bias=true, add_turn_bias=true, add_beta_decay=true, add_γ2=true, add_depletion_factor=true, kwargs...)
# run_beta_lik_stay_turn_decay_γ2(data; kwargs...) = run_beta_lik(
#     data; add_βleaf=false, add_stay_bias=true, add_turn_bias=true, add_beta_decay=true, add_γ2=true, kwargs...)

# (fn_name, fn, fn_add_leaf)
fns = [
    ("beta_lik_leaf_stay_turn_decay_γ2_depletion", run_beta_lik_leaf_stay_turn_decay_γ2_depletion, true),
    ("beta_lik_leaf_stay_turn_decay_γ2", run_beta_lik_leaf_stay_turn_decay_γ2, true),

    # No Leaf
    # ("beta_lik_stay_turn_decay_γ2_depletion", run_beta_lik_stay_turn_decay_γ2_depletion, true),
    # ("beta_lik_stay_turn_decay_γ2", run_beta_lik_stay_turn_decay_γ2, true),
]
base_dir = "../results/beta_bernoulli_depletion"
i = parse(Int, ARGS[1])
(fn_ind, animal_ind) = fldmod1(i, length(animals))
animal = animals[animal_ind]
data = load_animal(animal, ".."; depletion=true)
(fn_name, fn, fn_add_leaf) = fns[fn_ind]
flag_loocv = parse(Bool, ARGS[2])

@info animal
@info fn_name
@info flag_loocv

function run_fn(fn_name, fn, fn_add_leaf, rewscaled, delay_turn_bias, flag_loocv, full)
    # Create the base filename
    fname = fn_name
    fname = rewscaled ? fname * "_rewscaled" : fname
    fname = delay_turn_bias ? fname * "_delayturnbias" : fname
    fname *= "_$(animal)"
    fname = full ? fname * "_full" : fname
    @info fname
    
    if flag_loocv
        fname_loocv = fname * "_loocv"
    
        results = load("$(base_dir)/$(fname).jld2", "results")
        results_loocv = fn(data; extended=true, rewscaled, delay_turn_bias, loocv_data=results, full)
        save("$(base_dir)_loocv/$(fname_loocv).jld2", "results_loocv", results_loocv; compress=true)
    else
        results = fn(data; extended=true, rewscaled, delay_turn_bias, full)
        save("$(base_dir)/$(fname).jld2", "results", results; compress=true)
        write_EM_to_mat(results, "$(base_dir)/$(fname).mat"; rewscaled=rewscaled, delay_turn_bias=delay_turn_bias)
        Q = find_Q_vals_beta_lik(data, results; rewscaled, delay_turn_bias, add_leaf=fn_add_leaf);
        CSV.write("$(base_dir)/Q_vals_$(fname).csv.gz", Q; compress=true)
    end
end

run_fn(fn_name, fn, fn_add_leaf, false, false, flag_loocv, false)
# run_fn(fn_name, fn, fn_add_leaf, true, false, flag_loocv, false)