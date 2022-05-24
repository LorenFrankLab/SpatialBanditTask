using StatsPlots
using FileIO
using EM
include("../code/hmm.jl")
include("../code/em_scripts.jl")
include("../code/util.jl")

# number of contingencies
i = parse(Int, ARGS[1])
(fn_ind, animal_ind) = fldmod1(i, length(animals))
animal = animals[animal_ind]
data = load_animal(animal, "..")
println(animal)
println(fn_ind)
if fn_ind == 1
	results = run_hmm_leaf_stay_turn_leafspatial(data; ϕ=get_contingencies_base());
	save("../results/hmm_contingency/contingencies_base_$(animal).jld2", "contingencies_base_$(animal)", results; compress=true)
elseif fn_ind == 2
	results = run_hmm_leaf_stay_turn_leafspatial(data; ϕ=get_contingencies_observed());
	save("../results/hmm_contingency/contingencies_observed_$(animal).jld2", "contingencies_observed_$(animal)", results; compress=true)
else
	n = fn_ind - 2
	results = run_hmm_leaf_stay_turn_leafspatial(data; ϕ=get_contingencies(;n=n));
	save("../results/hmm_contingency/contingencies_$(n)_$(animal).jld2", "contingencies_$(n)_$(animal)", results; compress=true)
end
