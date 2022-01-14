# SpatialBanditTask
Modeling Alison's 6 arm/3 stem task. 

-----
Notes on Jan 14th, 2022 - Shijie Gu

### 1. Q learning
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This simple model-free learns the reward probability by moving average of reward history.

**Relevent code files**: 
`senor_byDay.ipynb` (the inital code from Natheniel when the project started).
`senor_by_session.ipynb` (adapting byDay to session)

### 2. Hidden Markov Model
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This model follows the POMDP narrative, infering the distribution over the states through the incremental inquiry to the environment.

**Relevent code files**: `hmm_model.m` (Estimate distribution, saves calculated point estimate and uncertainty/Shannon entropy),`plot_hmm.m`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; _The depletion version needs more charaterization (plot examples etc.)._

### 3. Comparing Q and HMM:

**Relevent code files**:
`fit_Q_state_space.ipynb` 
(fit a choice model from values estimates that are derived from either the Q learning model or HMM. Save parameters, likelihoods.)
`plot_model_comparison.m` (plot model comparison)

All other code files are used in these main scripts.
Behavior data are in the **data** folder.
Results from the code are saved in various folders.

-----
