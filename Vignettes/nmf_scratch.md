In this post, I will be discussing Non-negative Matrix Factorization
(NMF). NMF is a low-rank approximation algorithm that discovers latent
features in your data. It is similar to PCA in the sense that they both
reduce high-dimensional data into lower dimensions for better
understanding of the data. The major difference is that PCA projects
data onto a subspace that maximizes variability for the discovered
features, while NMF discovers non-negative features that are additive in
nature. NMF is formally defined as:

*V* ≈ *W**H*

where *V* is a non-negative matrix and both *W* and *H* are unique and
non-negative matrices. In other words, the matrix *V* is factorized into
two matrices *W* and *H*, where *W* is the features matrix or the basis
and *H* is the coefficient matrix. Typically, this means that *H*
represents a coordinate system that uses *W* to reconstruct *V*. We can
consider that *V* is a linear combination of column vectors of *W* using
the coordinate system in *H*, *v*<sub>*i*</sub> = *W**h*<sub>*i*</sub>.

Here, I will describe two algorithms to solve for NMF using iterative
updates of *W* and *H*. First, we will consider the cost function. A
cost function is a function that quantifies or measures the error
between the predicted values and the expected values. The Mean Squared
Error (MSE), or L2 loss is one of the most popular cost functions in
linear regressions. Given an linear equation *y* = *m**x* + *b*, MSE is:

$$MSE = \\frac{1}{N}\\Sigma\_{i=1}^{n}(y\_{i}-(mx\_{i}+b))^{2} $$

For the cost function in NMF, we use a similar function called the
Frobenius Norm. It is defined as:

$$||A||\_{F} = \\sqrt{\\Sigma\_{i=1}^{m}\\Sigma\_{j=1}^{n} |a\_{ij}|^{2}}$$

In the case of NMF, we are using the square of the Forbenius norm to
measure how good of an approximation *W**H* is for *V*.

||*V* − *W**H*||<sub>*F*</sub><sup>2</sup> = *Σ*<sub>*i*, *j*</sub>(*V* − *W**H*)<sub>*i**j*</sub><sup>2</sup>

We can see that as *W**H* approaches *V*, then the equation will slowly
converge to zero. Therefore, the optimization can be defined as the
following:

*M**i**n**i**m**i**z**e*||*V* − *W**H*||<sub>*F*</sub><sup>2</sup>*w**i**t**h**r**e**s**p**e**c**t**t**o**W**a**n**d**H*, *s**u**b**j**e**c**t**t**o**t**h**e**c**o**n**s**t**r**a**i**n**t**s**W*, *H* ≥ 0

In the paper by Lee & Seung, they introduced the multiplicative update
rule to solve for NMF. Please see their original paper for details on
the proof. Essentially the update causes the function value to be
non-increasing to converge to zero.

$$H\_{ik} \\leftarrow H\_{ik}\\frac{(W^{T}V)\_{ik}}{(W^{T}WH)\_{ik}}$$

$$W\_{kj} \\leftarrow W\_{kj}\\frac{(VH^{T})\_{kj}}{(WHH^{T})\_{kj}}$$

Here we will implement NMF using the multiplicative update rule. To get
started, make sure you have installed both
[R](https://www.r-project.org/) and [RStudio](https://rstudio.com/). In
addition, we will also be using the package
[NMF](https://cran.r-project.org/web/packages/NMF/index.html) to
benchmark our work.

Here I write a function to solve for NMF using the multiplicative update
rule. I added a `delta` variable to the denominator update rule to
prevent division by zero. `K` specifies the column length of *W* and the
row length of *H*. `K` can also be considered as the number of hidden
features we are discovering in *V*. `K` is less than *n* in a *n* × *m*
matrix. After iterating for `x` number of `steps`, the function returns
a `list` containing `W` and `H` for *W* and *H* respectively.

    nmf_mu <- function(R, K, delta = 0.001, steps = 5000){
      N = dim(R)[1]
      M = dim(R)[2]
      K = N
      W <- rmatrix(N,K) #Random initialization of W
      H <- rmatrix(K,M) #Random initialization of H
      for (step in 1:steps){
        W_TA = t(W) %*% R
        W_TWH = t(W) %*% W %*% H + delta
        for (i in 1:dim(H)[1]){
          for (j in 1:dim(H)[2]){
            H[i, j] = H[i, j] * W_TA[i, j] / W_TWH[i, j] #Updating H
          }
        }
        RH_T = R %*% t(H)
        WHH_T = W %*% H %*% t(H) + delta
        for (i in 1:dim(W)[1]){
          for (j in 1:dim(W)[2]){
            W[i, j] = W[i, j] * RH_T[i, j] / WHH_T[i, j] #Updating W
          }
        }
      }
      list <- list('W'=W, 'H'=H)
      return(list)
    }

Let’s initialize a random *n* × *m* matrix and test our function.

    require(NMF)

    ## Loading required package: NMF

    ## Loading required package: pkgmaker

    ## Loading required package: registry

    ## Loading required package: rngtools

    ## Loading required package: cluster

    ## NMF - BioConductor layer [OK] | Shared memory capabilities [NO: bigmemory] | Cores 3/4

    ##   To enable shared memory capabilities, try: install.extras('
    ## NMF
    ## ')

    R <- rmatrix(5,6)
    R

    ##            [,1]       [,2]      [,3]       [,4]       [,5]      [,6]
    ## [1,] 0.38590816 0.07524472 0.3840033 0.71850549 0.94777199 0.2569990
    ## [2,] 0.46994229 0.01347989 0.6568133 0.74398321 0.47960622 0.1895243
    ## [3,] 0.09009019 0.16339225 0.2261623 0.02087745 0.85048408 0.2473095
    ## [4,] 0.89357384 0.39553503 0.6977186 0.08057693 0.05300029 0.5915455
    ## [5,] 0.86357834 0.66435474 0.6247102 0.35868982 0.54430141 0.5297718

    nmf_mu_results <- nmf_mu(R)
    cat('\fMatrix W is:\n')

    ## Matrix W is:

    print(nmf_mu_results$W)

    ##             [,1]         [,2]         [,3]         [,4]         [,5]
    ## [1,] 0.109275756 1.000646e+00 0.5594300085 0.0004456571 7.725333e-02
    ## [2,] 0.006772038 1.001991e-01 1.0447507882 0.0020328963 7.822121e-02
    ## [3,] 0.126058950 5.267083e-02 0.0000493097 0.0014999231 9.083227e-01
    ## [4,] 0.641394480 1.348717e-13 0.1091054404 1.1144080468 6.336317e-06
    ## [5,] 1.020070670 4.522571e-02 0.4899339297 0.0112129823 3.633405e-01

    cat('Matrix H is:\n')

    ## Matrix H is:

    print(nmf_mu_results$H)

    ##             [,1]         [,2]         [,3]         [,4]         [,5]      [,6]
    ## [1,] 0.626070275 6.167153e-01 0.2383297795 9.686670e-03 2.828901e-02 0.3619009
    ## [2,] 0.071086355 4.167417e-14 0.0009175065 3.362347e-01 6.926071e-01 0.1153150
    ## [3,] 0.437429496 5.074458e-04 0.6092903342 6.794784e-01 3.252372e-01 0.1510770
    ## [4,] 0.397882928 4.023670e-05 0.4284747916 6.223050e-19 2.795722e-26 0.3069356
    ## [5,] 0.007246151 9.372761e-02 0.2141528579 6.599956e-04 8.915154e-01 0.2138651

Let’s see if we can reconstruct our original matrix and compare it to
the `nmf` function.

    R

    ##            [,1]       [,2]      [,3]       [,4]       [,5]      [,6]
    ## [1,] 0.38590816 0.07524472 0.3840033 0.71850549 0.94777199 0.2569990
    ## [2,] 0.46994229 0.01347989 0.6568133 0.74398321 0.47960622 0.1895243
    ## [3,] 0.09009019 0.16339225 0.2261623 0.02087745 0.85048408 0.2473095
    ## [4,] 0.89357384 0.39553503 0.6977186 0.08057693 0.05300029 0.5915455
    ## [5,] 0.86357834 0.66435474 0.6247102 0.35868982 0.54430141 0.5297718

    nmf_mu_results$W %*% nmf_mu_results$H

    ##            [,1]       [,2]      [,3]       [,4]       [,5]      [,6]
    ## [1,] 0.38499485 0.07491669 0.3845520 0.71768189 0.94696557 0.2561121
    ## [2,] 0.46974303 0.01203814 0.6558848 0.74369324 0.47911739 0.1891958
    ## [3,] 0.08986615 0.16287748 0.2252846 0.01956385 0.84984603 0.2464209
    ## [4,] 0.89268794 0.39565856 0.6968374 0.08034777 0.05363521 0.5906574
    ## [5,] 0.86325668 0.66339724 0.6242816 0.35822687 0.54344878 0.5295449

We get the same results using the `nmf` function with the `lee` method.

    nmf <- nmf(R, dim(R)[1], method = 'lee')
    basis(nmf) %*% coefficients(nmf)

    ##            [,1]       [,2]      [,3]       [,4]      [,5]      [,6]
    ## [1,] 0.38515785 0.07671845 0.3862359 0.71705144 0.9484342 0.2557344
    ## [2,] 0.47042756 0.01108820 0.6552286 0.74517053 0.4785469 0.1920720
    ## [3,] 0.08994887 0.16375738 0.2264305 0.02581819 0.8503109 0.2469080
    ## [4,] 0.89349535 0.39599817 0.6982080 0.08030798 0.0530306 0.5909305
    ## [5,] 0.86378851 0.66387352 0.6243622 0.35887046 0.5443661 0.5303499

Now we will take a look at another method of implementing NMF. This one
is called Stochastic Gradient Descent (SGD). A gradient descent is a
first-order iterative optimization algorithm to finding a local minimum
for a function that is differentiable. In fact, we used the Block
Coordinate Descent in the multiplicative update rule. In SGD, we take
the derivative of the cost function like before. However, we will now be
focusing on taking the derivative of each variable, setting them to zero
or lower, solving for the feature variables, and finally updating each
feature. We also add a regularization term in the cost function to
control for overfitting.

||*V* − *W**H*||<sub>*F*</sub><sup>2</sup> = *Σ*<sub>*i*, *j*</sub>(*V* − *W**H*)<sub>*i**j*</sub><sup>2</sup>

*e*<sub>*i**j*</sub><sup>2</sup> = *Σ*<sub>*i*, *j*</sub>(*v*<sub>*i**j*</sub> − *v̂*<sub>*i**j*</sub>)<sup>2</sup> = (*v*<sub>*i**j*</sub> − *Σ*<sub>*k* = 1</sub><sup>*K*</sup>*w*<sub>*i**k*</sub>*h*<sub>*k**j*</sub>)<sup>2</sup>

*e*<sub>*i**j*</sub><sup>2</sup> = (*v*<sub>*i**j*</sub> − *Σ*<sub>*k* = 1</sub><sup>*K*</sup>*w*<sub>*i**k*</sub>*h*<sub>*k**j*</sub>)<sup>2</sup> + *λ**Σ*<sub>*k* = 1</sub><sup>*K*</sup>(||*W*||<sup>2</sup> + ||*H*||<sup>2</sup>)

*λ* is used to control the magnitudes of *w* and *h* such that they
would provide a good approximation of *v*. We will update each feature
with each sample. We choose a small *λ*, such as 0.01. The update is
given by the equations below:

$$w\_{ik} \\leftarrow w\_{ik} - \\eta \\frac{\\partial}{\\partial w\_{ik}}e\_{ij}^{2}$$

$$h\_{kj} \\leftarrow h\_{kj} - \\eta \\frac{\\partial}{\\partial h\_{kj}}e\_{ij}^{2}$$

*η* is the learning rate and modifies the magnitude that we update the
features. We first solve for $ e\_{ij}^{2}$.

$$Using the chain rule, \\frac{\\partial}{\\partial h\_{kj}}(v\_{ij} - \\Sigma\_{k=1}^{K}w\_{ik}h\_{kj}) = \\frac{\\partial u^{2}}{\\partial v} \\frac{\\partial u}{\\partial v}, where u = (v\_{ij} - \\Sigma\_{k=1}^{K}w\_{ik}h\_{kj}) and \\frac{\\partial u^{2}}{\\partial v} = 2u$$

$$ \\frac{\\partial}{\\partial h\_{kj}}e\_{ij}^{2} = 2(v\_{ij} - \\Sigma\_{k=1}^{K}w\_{ik}h\_{kj}) \\frac{\\partial}{\\partial h\_{kj}}(v\_{ij} - \\Sigma\_{k=1}^{K}w\_{ik}h\_{kj}) + 2\\lambda h\_{kj} $$

$$ \\frac{\\partial}{\\partial h\_{kj}}e\_{ij}^{2} = -2e\_{ij}w\_{ik} + 2\\lambda h\_{kj} $$

The final update rules for both *W* and *H*:

$$ \\frac{\\partial}{\\partial h\_{kj}}e\_{ij}^{2} = -2e\_{ij}w\_{ik} + 2\\lambda h\_{kj} $$

$$ \\frac{\\partial}{\\partial w\_{ik}}e\_{ij}^{2} = -2e\_{ij}h + 2\\lambda w\_{ik} $$

    frob_norm <- function(M){
      norm = 0
      for (i in 1:dim(M)[1]){
        for (j in 1:dim(M)[2]){
          norm = norm + M[i,j] ** 2
        }
      }
      return(sqrt(norm))
    }
    nmf_sgd <- function(A,steps = 50000, lam = 1e-2, lr = 1e-3){
      N = dim(A)[1]
      M = dim(A)[2]
      K = N
      W <- rmatrix(N,K)
      H <- rmatrix(K,M)
      for (step in 1:steps){
        R =  A - W %*% H 
        dW = R %*% t(H) - W*lam
        dH = t(W) %*% R - H*lam
        W = W + lr * dW
        H = H + lr * dH
        if (frob_norm(A - W %*% H) < 0.01){
          print(frob_norm(A - W %*% H))
          break
        }
      }
      list <- list(W, t(H))
      return(list)
    }

    nmf_sgd_results <- nmf_sgd(R)
    R

    ##            [,1]       [,2]      [,3]       [,4]       [,5]      [,6]
    ## [1,] 0.38590816 0.07524472 0.3840033 0.71850549 0.94777199 0.2569990
    ## [2,] 0.46994229 0.01347989 0.6568133 0.74398321 0.47960622 0.1895243
    ## [3,] 0.09009019 0.16339225 0.2261623 0.02087745 0.85048408 0.2473095
    ## [4,] 0.89357384 0.39553503 0.6977186 0.08057693 0.05300029 0.5915455
    ## [5,] 0.86357834 0.66435474 0.6247102 0.35868982 0.54430141 0.5297718

The reconstructed method using our NMF function looks like this:

    nmf_sgd_results[[1]] %*% t(nmf_sgd_results[[2]])

    ##            [,1]       [,2]      [,3]       [,4]       [,5]      [,6]
    ## [1,] 0.38305104 0.07814005 0.3868637 0.71353756 0.94217127 0.2547610
    ## [2,] 0.47031999 0.01478160 0.6501358 0.73856920 0.47951437 0.1916646
    ## [3,] 0.09264363 0.16279282 0.2239151 0.02513887 0.84299766 0.2458078
    ## [4,] 0.88730738 0.39710504 0.6943744 0.08319123 0.05491914 0.5861343
    ## [5,] 0.86052368 0.65563023 0.6234492 0.35693779 0.54306292 0.5292586

As you can see, it is a near approximation of the original matrix. In
another post, I will go into detail the differences between each
dimensional reduction technique. See below for more information on NMF.

Sources and additional information:

<a href="https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization.pdf" class="uri">https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization.pdf</a>

<a href="https://github.com/hpark3910/nonnegative-matrix-factorization" class="uri">https://github.com/hpark3910/nonnegative-matrix-factorization</a>

<a href="https://www.almoststochastic.com/2013/06/nonnegative-matrix-factorization.html" class="uri">https://www.almoststochastic.com/2013/06/nonnegative-matrix-factorization.html</a>

<a href="https://github.com/carriexu24/NMF-from-scratch-using-SGD" class="uri">https://github.com/carriexu24/NMF-from-scratch-using-SGD</a>

<a href="https://albertauyeung.github.io/2017/04/23/python-matrix-factorization.html" class="uri">https://albertauyeung.github.io/2017/04/23/python-matrix-factorization.html</a>
