## Structure:
- /EM: Package for expectation maximization
- /code: learning models
- /data: animal data
- /notebooks_metalearning: analysis for metalearning paper
- /results: model results

## Getting Started:

Clone the repo:
- `git clone https://github.com/LorenFrankLab/SpatialBanditTask`

The analyses here depend on a custom package for expectation maximizaton.
- `julia -e 'using Pkg; Pkg.add(path="$(pwd())/EM")'`

For "Meta-learning is expressed through altered prefrontal cortical dynamics", see `notebooks_metalearning/beta_model.ipynb`