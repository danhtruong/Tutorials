In this post, we will extend what we did in the previous PCA tutorial by
doing it from scratch in R. We will be focusing on Singular Value
Decomposition (SVD), which is a tool for matrix factorization. SVD
states that any matrix $M A^{m n} $ can be decomposed into the
following:

*M* = *U**Σ**V*<sup>*T*</sup>

where *U* is an *m* × *m* unitary matrix, *Σ* is a diagonal *m* × *n*
matrix with non-negative real numbers on the diagonal, and *V* is an
*n* × *n* unitary matrix. If *M* is real, then *U* and *V* are
orthogonal matrices, or uncorrelated. *U* and *V* are equal to the
eigenvectors of *M**M*<sup>*t*</sup> and *M*<sup>*t*</sup>*M*,
respectively. The entries in *Σ* are singular values of *M*. The number
of non-zero singular values is equal to the rank of *M*. If you recall,
the rank of a matrix is defined as the maximum number of linearly
independent columns in the matrix. As it relates to PCA, these are the
nummber of Principal Components.

Performing SVD on the covariance matrix of *M* is essentially PCA. We
will be using the Iris dataset.

    data(iris)
    D <- data.matrix(iris[,c(1:4)]) #Here we are only taking the first four columns as the last one is categorical data.

The covariance is defined as follows:

$cov\_{x,y} = \\frac{\\Sigma(x\_{i}-\\overline{x})(y\_{i}-\\overline{y})}{N-1}$

First, we will center our data by subtracting each value by the of their
respective column.

    xcenter = colMeans(D) #Find the mean of the column
    print(xcenter)

    ## Sepal.Length  Sepal.Width Petal.Length  Petal.Width 
    ##     5.843333     3.057333     3.758000     1.199333

    D_centered <- D - rep(xcenter, rep.int(nrow(D), ncol(D)))
    head(D_centered)

    ##      Sepal.Length Sepal.Width Petal.Length Petal.Width
    ## [1,]   -0.7433333  0.44266667       -2.358  -0.9993333
    ## [2,]   -0.9433333 -0.05733333       -2.358  -0.9993333
    ## [3,]   -1.1433333  0.14266667       -2.458  -0.9993333
    ## [4,]   -1.2433333  0.04266667       -2.258  -0.9993333
    ## [5,]   -0.8433333  0.54266667       -2.358  -0.9993333
    ## [6,]   -0.4433333  0.84266667       -2.058  -0.7993333

We will multiply the transposed form of our matrix by our matrix
followed dividing by *N* − 1, where N is number of rows. We will compare
this to the R function `cov()`

    covariance = t(D_centered) %*% (D_centered)
    covariance = covariance / (dim(D_centered)[1] - 1)
    covariance

    ##              Sepal.Length Sepal.Width Petal.Length Petal.Width
    ## Sepal.Length    0.6856935  -0.0424340    1.2743154   0.5162707
    ## Sepal.Width    -0.0424340   0.1899794   -0.3296564  -0.1216394
    ## Petal.Length    1.2743154  -0.3296564    3.1162779   1.2956094
    ## Petal.Width     0.5162707  -0.1216394    1.2956094   0.5810063

    cov(D_centered)

    ##              Sepal.Length Sepal.Width Petal.Length Petal.Width
    ## Sepal.Length    0.6856935  -0.0424340    1.2743154   0.5162707
    ## Sepal.Width    -0.0424340   0.1899794   -0.3296564  -0.1216394
    ## Petal.Length    1.2743154  -0.3296564    3.1162779   1.2956094
    ## Petal.Width     0.5162707  -0.1216394    1.2956094   0.5810063

    cat('\n Are the values the same:', all(round(cov(D_centered) - covariance, 3) == 0)) 

    ## 
    ##  Are the values the same: TRUE

Next, we will solve for *M**M*<sup>*t*</sup> and *M*<sup>*t*</sup>*M* to
find *U* and *V*.

    MTM <- t(covariance) %*% covariance
    MTM.e <- eigen(MTM)
    V <- MTM.e$vectors
    V

    ##             [,1]        [,2]        [,3]       [,4]
    ## [1,]  0.36138659 -0.65658877 -0.58202985  0.3154872
    ## [2,] -0.08452251 -0.73016143  0.59791083 -0.3197231
    ## [3,]  0.85667061  0.17337266  0.07623608 -0.4798390
    ## [4,]  0.35828920  0.07548102  0.54583143  0.7536574

    MMT <- covariance %*% t(covariance)
    MMT.e <- eigen(MMT)
    U <- MMT.e$vectors
    U

    ##             [,1]        [,2]        [,3]       [,4]
    ## [1,]  0.36138659 -0.65658877 -0.58202985  0.3154872
    ## [2,] -0.08452251 -0.73016143  0.59791083 -0.3197231
    ## [3,]  0.85667061  0.17337266  0.07623608 -0.4798390
    ## [4,]  0.35828920  0.07548102  0.54583143  0.7536574

Note that here we find that *U* and *V* are equivalent. This is a
special case because we are using the covariance matrix. Now we find the
singular values *Σ*, which are the square roots of the non-zero
eigenvalues of *M**M*<sup>*t*</sup> and *M*<sup>*t*</sup>*M*.

    sigma <- sqrt(MMT.e$values)
    sigma <- sigma * diag(length(sigma))
    sigma

    ##          [,1]      [,2]      [,3]       [,4]
    ## [1,] 4.228242 0.0000000 0.0000000 0.00000000
    ## [2,] 0.000000 0.2426707 0.0000000 0.00000000
    ## [3,] 0.000000 0.0000000 0.0782095 0.00000000
    ## [4,] 0.000000 0.0000000 0.0000000 0.02383509

Now recall the equation for SVD.

*M* = *U**Σ**V*<sup>*T*</sup>

Now that, we have the decommposed values, we can recompose them using
matrix multiplcation to recover our covariance matrix of the Iris
dataset.

    M <- U %*% sigma %*% t(V)
    M

    ##            [,1]       [,2]       [,3]       [,4]
    ## [1,]  0.6856935 -0.0424340  1.2743154  0.5162707
    ## [2,] -0.0424340  0.1899794 -0.3296564 -0.1216394
    ## [3,]  1.2743154 -0.3296564  3.1162779  1.2956094
    ## [4,]  0.5162707 -0.1216394  1.2956094  0.5810063

    covariance

    ##              Sepal.Length Sepal.Width Petal.Length Petal.Width
    ## Sepal.Length    0.6856935  -0.0424340    1.2743154   0.5162707
    ## Sepal.Width    -0.0424340   0.1899794   -0.3296564  -0.1216394
    ## Petal.Length    1.2743154  -0.3296564    3.1162779   1.2956094
    ## Petal.Width     0.5162707  -0.1216394    1.2956094   0.5810063

    cat('\n Are the values the same:', all(round(M - covariance, 3) == 0)) 

    ## 
    ##  Are the values the same: TRUE

PCA is an orthoganol projection.

    pc <- D_centered %*% V
    head(pc)

    ##           [,1]       [,2]        [,3]         [,4]
    ## [1,] -2.684126 -0.3193972 -0.02791483  0.002262437
    ## [2,] -2.714142  0.1770012 -0.21046427  0.099026550
    ## [3,] -2.888991  0.1449494  0.01790026  0.019968390
    ## [4,] -2.745343  0.3182990  0.03155937 -0.075575817
    ## [5,] -2.728717 -0.3267545  0.09007924 -0.061258593
    ## [6,] -2.280860 -0.7413304  0.16867766 -0.024200858

    pca_prcomp <- prcomp(D)
    cat('\n Comparing our calculations to the R function prcommp:\n\n')

    ## 
    ##  Comparing our calculations to the R function prcommp:

    cat('\n Do we get the same rotation matrix:',  all(round(pca_prcomp$rotation - V, 3) == 0)) 

    ## 
    ##  Do we get the same rotation matrix: FALSE

    cat('\n Are the principal components the same:', all(round(pca_prcomp$X - pc, 3) == 0)) 

    ## 
    ##  Are the principal components the same: TRUE

It looks like something is wrong with our rotation matrix. Upon closer
inspection, it appears the third column has the opposite sign, which is
a multiple of  − 1. Recall that *A**u* = *λ**u*, where *A* is a vector,
*u* is the eigenvector, and *l**a**m**b**d**a* is the eigenvalue.

If *A**u* = *λ**u*, then $A(ku) = Akv = k u = (ku) $. Therefore, any
multiple of *u* is also an eigenvector of *A*.

    V

    ##             [,1]        [,2]        [,3]       [,4]
    ## [1,]  0.36138659 -0.65658877 -0.58202985  0.3154872
    ## [2,] -0.08452251 -0.73016143  0.59791083 -0.3197231
    ## [3,]  0.85667061  0.17337266  0.07623608 -0.4798390
    ## [4,]  0.35828920  0.07548102  0.54583143  0.7536574

    pca_prcomp$rotation

    ##                      PC1         PC2         PC3        PC4
    ## Sepal.Length  0.36138659 -0.65658877  0.58202985  0.3154872
    ## Sepal.Width  -0.08452251 -0.73016143 -0.59791083 -0.3197231
    ## Petal.Length  0.85667061  0.17337266 -0.07623608 -0.4798390
    ## Petal.Width   0.35828920  0.07548102 -0.54583143  0.7536574
