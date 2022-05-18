using StatsPlots
using FileIO
using EM
include("../code/qlearner.jl")
include("../code/util.jl")
include("../code/em_scripts.jl")

fns = [
("q_leaf", run_q_leaf),
("q_leaf_γ2", run_q_leaf_γ2),
("q_leaf_retainbelief", run_q_leaf_retainbelief),
("q_leaf_stay", run_q_leaf_stay),
("q_leaf_stay_γ2", run_q_leaf_stay_γ2),
("q_leaf_stay_retainbelief", run_q_leaf_stay_retainbelief),

("q_leaf_stay_turn", run_q_leaf_stay_turn),
("q_leaf_stay_spatial", run_q_leaf_stay_spatial),
("q_leaf_stay_turn_leafspatial", run_q_leaf_stay_turn_leafspatial),
("q_leaf_stay_turn_leafturn", run_q_leaf_stay_turn_leafturn),
("q_leaf_stay_spatial_leafspatial", run_q_leaf_stay_spatial_leafspatial),
("q_leaf_stay_spatial_leafturn", run_q_leaf_stay_spatial_leafturn),

("q_leaf_stay_turn_γ2", run_q_leaf_stay_turn_γ2),
("q_leaf_stay_spatial_γ2", run_q_leaf_stay_spatial_γ2),
("q_leaf_stay_turn_leafspatial_γ2", run_q_leaf_stay_turn_leafspatial_γ2),
("q_leaf_stay_turn_leafturn_γ2", run_q_leaf_stay_turn_leafturn_γ2),
("q_leaf_stay_spatial_leafspatial_γ2", run_q_leaf_stay_spatial_leafspatial_γ2),
("q_leaf_stay_spatial_leafturn_γ2", run_q_leaf_stay_spatial_leafturn_γ2),
]
i = parse(Int, ARGS[1])
(fn_ind, animal_ind) = fldmod1(i, length(animals))
animal = animals[animal_ind]
data = load_animal(animal, ".."; depletion=false)
(fname, fn) = fns[fn_ind]

println(animal)
println(fn)

results = fn(data; extended=true)
save("../results/q_biases/$(fname)_$(animal).jld2", "$(fname)_$(animal)", results)

results = fn(data; extended=true, rewscaled=true)
save("../results/q_biases/$(fname)_rewscaled_$(animal).jld2", "$(fname)_rewscaled_$(animal)", results)
