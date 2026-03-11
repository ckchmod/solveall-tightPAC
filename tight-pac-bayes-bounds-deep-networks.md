## Tight PAC-Bayes Bounds for Deep Neural Networks

### Setup

Let \(\mathcal{X}\) be an input space, \(\mathcal{Y}=\{1,\dots,C\}\) a finite label set, and \(\mathcal{D}\) an unknown distribution on \(\mathcal{X}\times\mathcal{Y}\). A training sample is
\[
S=((x_1,y_1),\dots,(x_n,y_n))\sim \mathcal{D}^n.
\]

Notation used throughout:

- Deterministic risk: \(L_{\mathcal D}(h_\theta)=\mathbb E_{(x,y)\sim\mathcal D}[\ell(h_\theta(x),y)]\).
- Deterministic empirical risk: \(\hat L_S(h_\theta)=\frac1n\sum_{i=1}^n\ell(h_\theta(x_i),y_i)\).
- Gibbs population risk: \(L_{\mathcal D}(Q):=\mathbb E_{\theta\sim Q}[L_{\mathcal D}(h_\theta)]\).
- Gibbs empirical risk: \(\hat L_S(Q):=\mathbb E_{\theta\sim Q}[\hat L_S(h_\theta)]\).
- Binary relative entropy: \(\mathrm{kl}(p\|q)=p\ln\frac pq+(1-p)\ln\frac{1-p}{1-q}\), with standard continuous extensions at \(p\in\{0,1\}\).

If a data-dependent prior is used, legality is by explicit sample splitting: \(S=S_0\cup S_1\), disjoint, with the PAC-Bayes generalization event taken over \(S_1\sim\mathcal D^{n_1}\) and sample-size constants using \(n_1\).

Fix a deep neural-network architecture with parameter vector \(\theta\in\mathbb{R}^p\) (typically \(p\gg n\)), and let \(h_\theta:\mathcal{X}\to\mathcal{Y}\) be the induced classifier. For bounded loss \(\ell:\mathcal{Y}\times\mathcal{Y}\to[0,1]\) (especially \(0\)-\(1\) loss), define
\[
L_{\mathcal D}(h_\theta)=\mathbb E_{(x,y)\sim\mathcal D}[\ell(h_\theta(x),y)],
\qquad
\hat L_S(h_\theta)=\frac1n\sum_{i=1}^n \ell(h_\theta(x_i),y_i).
\]

A stochastic neural network (Gibbs predictor) is specified by posterior \(Q\) on \(\mathbb{R}^p\); prediction uses \(\theta\sim Q\). Its risks are
\[
L_{\mathcal D}(Q)
\quad\text{and}\quad
\hat L_S(Q).
\]

Let \(P\) be a prior on \(\mathbb{R}^p\), admissible for the chosen mechanism (independent of the PAC-Bayes sample, or sample-splitting legal), and let \(\mathrm{KL}(Q\|P)\) be the Kullback-Leibler divergence.

### Open Problem

Determine whether there exists a PAC-Bayes framework (choice of admissible priors/posteriors and bound/estimation procedure) such that for modern large-scale deep networks and datasets one can compute in polynomial time a certified upper bound \(B(S,\delta)\) satisfying, with probability at least \(1-\delta\) over the PAC-Bayes sample (\(S\sim\mathcal D^n\) in the independent-prior case, or \(S_1\sim\mathcal D^{n_1}\) in sample-splitting),
\[
L_{\mathcal D}(Q)
\le
\hat L_S(Q)
+\text{PAC-Bayes complexity term derived from a chosen }\mathrm{kl}\text{-bound with explicit }C(n_{\mathrm{eff}},\delta)
\le B(S,\delta),
\]
where \(n_{\mathrm{eff}}\in\{n,n_1\}\) matches the legal mechanism, and the exact complexity expression is fixed by the selected theorem variant. (Any square-root relaxation must be explicitly derived from the chosen \(\mathrm{kl}\)-bound.)

The goal is to do so while simultaneously achieving:

1. **Non-vacuity at practical scale**: for standard trained architectures (including modern benchmark-scale settings), \(B(S,\delta)<1\) for \(0\)-\(1\) loss.
2. **Computational tractability**: the certificate is computable to certified numerical accuracy in time polynomial in explicit natural parameters (at least \(n_{\mathrm{eff}}\), \(p\), \(\log(1/\delta)\), and inverse numerical tolerance), under a stated computational model.
3. **Predictive tightness across hyperparameters**: over a hyperparameter family \(\Lambda\), \(\lambda\mapsto B_\lambda(S,\delta)\) is strongly positively associated with \(\lambda\mapsto \mathbb E_{\theta\sim Q_\lambda}[L_{\mathcal D}(h_\theta)]\).

### Why This Matters

A framework satisfying all three properties above would materially improve theoretical understanding and practical model selection for overparameterized deep learning.

### Key References

1. Dziugaite & Roy (2017), *Computing nonvacuous generalization bounds for deep (stochastic) neural networks with many more parameters than training data*.
2. McAllester (2003), *PAC-Bayesian stochastic model selection*.
3. Guedj (2019), *A Primer on PAC-Bayesian Learning*.
4. McAllester (1998), *Some PAC-Bayesian Theorems*.
5. McAllester (1999), *PAC-Bayesian Model Averaging*.
6. Arora et al. (2018), *Stronger generalization bounds for deep nets via a compression approach*.
7. Zhou et al. (2019), *Non-vacuous Generalization Bounds at the ImageNet Scale: a PAC-Bayesian Compression Approach*.
