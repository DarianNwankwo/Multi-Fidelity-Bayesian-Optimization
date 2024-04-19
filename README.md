# Multi-Fidelity Bayesian Optimization

## Code Design and Architecture
### High Level Overview of Bayesian Optimization
Bayesian Optimization (BO) is a technique for optimizing an objective function $f$ which is typically black-box. It leverages known observations to construct a statistical model of the underlying objective. This model is then used to construct heuristics that guide our sampling strategy.

BO can be expressed succinctly in 7 fundamental steps:
1. Gather initial samples
2. Initialize the surrogate model
3. Construct the acquisition function $\alpha(x)$
4. Optimize the acquisition function $x^* = \arg\max_x \alpha(x)$
5. Sample new data at $x^*$ and update surrogate
6. Repeat until the budget is exhausted
7. Make final recommendation $x^{final}$

### Induced Dependencies
The algorithm above naturally induces some dependencies. The code is seperated into distinct interellated chunks
that enable Bayesian Optimization. We provide a high-level overview of each.

#### Source Code Explanations
`kernels.jl`
Our kernel objects have several fields that allow us to query the hyperparamters, the evaluation
of the kernel and all of it's partials, including the partials with respect to our hyperparameters.
We leverage automatic differentiation to compute these partials and the `KernelGeneric` function
is responsible for facilitating the construction of a `Kernel` that abides by this specification.

Kernels have several properties that are worth writing support for, but we've only implemented
3 of the many, i.e. kernel addition, kernel scaling, and kernel multiplication. In order to
construct arbitrary kernels from arbitrary operations, we express our kernel objects as nodes in an
expression tree.

`surrogates.jl`



## Objectives
- [ ] Implement logic for computing the gradient of the log likelihood
- [ ] Write support for updating/conditioning the Gaussian Process on new observations
- [ ] Update abstract type dependencies in `surrogates.jl`

## Relevant Resources
- [Popular Acquisition Functions in Bayesian Optimization](https://ekamperi.github.io/machine%20learning/2021/06/11/acquisition-functions.html#upper-confidence-bound-ucb)
- [Write-Up on Thoughts Concerning Multifidelity Bayesian Optimization](https://www.overleaf.com/project/65d9077272a931242684d11f)
- [Interactive Guide on Gaussian Processes](https://infallible-thompson-49de36.netlify.app/#section-3)
- [Properties of Kernels](https://people.eecs.berkeley.edu/~jordan/kernels/0521813972c03_p47-84.pdf)
