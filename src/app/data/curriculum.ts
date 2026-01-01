export interface Resource {
  title: string;
  url: string;
  type: 'video' | 'article' | 'tutorial' | 'documentation' | 'course';
  source: string;
}

export interface Day {
  day: number;
  title: string;
  description: string;
  topics: string[];
  objectives: string[];
  resources: Resource[];
  practice: string[];
  timeEstimate: string;
}

export interface Week {
  week: number;
  title: string;
  days: Day[];
}

export interface Month {
  month: number;
  title: string;
  weeks: Week[];
}

export const curriculum: Month[] = [
  {
    month: 1,
    title: "Mathematics Foundations",
    weeks: [
      {
        week: 1,
        title: "Linear Algebra Basics",
        days: [
          {
            day: 1,
            title: "Vectors and Vector Spaces",
            description: "Introduction to vectors, vector operations, and vector spaces",
            topics: ["Vectors", "Scalar Multiplication", "Vector Addition", "Vector Spaces"],
            objectives: [
              "Understand what vectors are and how they represent data in AI",
              "Learn vector operations (addition, subtraction, scalar multiplication)",
              "Understand vector spaces and subspaces",
              "Apply vector concepts to real-world AI problems"
            ],
            resources: [
              {
                title: "3Blue1Brown - Vectors | Essence of Linear Algebra",
                url: "https://www.youtube.com/watch?v=fNk_zzaMoSs",
                type: "video",
                source: "YouTube"
              },
              {
                title: "Khan Academy - Vectors and Spaces",
                url: "https://www.khanacademy.org/math/linear-algebra/vectors-and-spaces",
                type: "course",
                source: "Khan Academy"
              },
              {
                title: "Linear Algebra for Machine Learning",
                url: "https://machinelearningmastery.com/linear-algebra-machine-learning/",
                type: "article",
                source: "Machine Learning Mastery"
              }
            ],
            practice: [
              "Solve 10 vector addition and subtraction problems",
              "Calculate dot products for various vector pairs",
              "Implement vector operations in Python using NumPy",
              "Visualize 2D and 3D vectors using matplotlib"
            ],
            timeEstimate: "4-6 hours"
          },
          {
            day: 2,
            title: "Matrices and Matrix Operations",
            description: "Learn about matrices, matrix multiplication, and their properties",
            topics: ["Matrices", "Matrix Multiplication", "Transpose", "Identity Matrix"],
            objectives: [
              "Understand matrix representation and notation",
              "Master matrix multiplication and its properties",
              "Learn about special matrices (identity, diagonal, symmetric)",
              "Understand how matrices transform data in neural networks"
            ],
            resources: [
              {
                title: "3Blue1Brown - Linear Transformations and Matrices",
                url: "https://www.youtube.com/watch?v=kYB8IZa5AuE",
                type: "video",
                source: "YouTube"
              },
              {
                title: "Khan Academy - Matrix Transformations",
                url: "https://www.khanacademy.org/math/linear-algebra/matrix-transformations",
                type: "course",
                source: "Khan Academy"
              },
              {
                title: "NumPy Matrix Tutorial",
                url: "https://numpy.org/doc/stable/user/tutorial-svd.html",
                type: "documentation",
                source: "NumPy Docs"
              }
            ],
            practice: [
              "Perform 15 matrix multiplication problems by hand",
              "Implement matrix operations in Python",
              "Create a matrix multiplication visualization",
              "Apply matrix transformations to rotate and scale 2D shapes"
            ],
            timeEstimate: "5-7 hours"
          },
          {
            day: 3,
            title: "Determinants and Inverses",
            description: "Understanding determinants, matrix inverses, and their applications",
            topics: ["Determinants", "Matrix Inverse", "Singular Matrices", "Applications"],
            objectives: [
              "Calculate determinants of 2x2 and 3x3 matrices",
              "Understand the geometric meaning of determinants",
              "Learn how to find matrix inverses",
              "Identify when matrices are invertible"
            ],
            resources: [
              {
                title: "3Blue1Brown - The Determinant",
                url: "https://www.youtube.com/watch?v=Ip3X9LOh2dk",
                type: "video",
                source: "YouTube"
              },
              {
                title: "Khan Academy - Determinants",
                url: "https://www.khanacademy.org/math/linear-algebra/matrix-transformations/determinant-depth/v/linear-algebra-determinant-when-row-multiplied-by-scalar",
                type: "course",
                source: "Khan Academy"
              },
              {
                title: "Matrix Inverse in Machine Learning",
                url: "https://machinelearningmastery.com/solve-linear-regression-using-linear-algebra/",
                type: "article",
                source: "ML Mastery"
              }
            ],
            practice: [
              "Calculate determinants for 10 different matrices",
              "Find inverses of 5 invertible matrices",
              "Implement determinant calculation in Python",
              "Solve systems of linear equations using matrix inverses"
            ],
            timeEstimate: "4-6 hours"
          },
          {
            day: 4,
            title: "Eigenvalues and Eigenvectors",
            description: "Understanding eigenvalues, eigenvectors and their significance in AI",
            topics: ["Eigenvalues", "Eigenvectors", "Characteristic Equation", "Diagonalization"],
            objectives: [
              "Understand what eigenvectors and eigenvalues represent",
              "Learn how to calculate eigenvalues and eigenvectors",
              "Understand their role in PCA and dimensionality reduction",
              "Apply eigen-decomposition to real problems"
            ],
            resources: [
              {
                title: "3Blue1Brown - Eigenvectors and Eigenvalues",
                url: "https://www.youtube.com/watch?v=PFDu9oVAE-g",
                type: "video",
                source: "YouTube"
              },
              {
                title: "StatQuest - Eigenvalues and Eigenvectors",
                url: "https://www.youtube.com/watch?v=FgakZw6K1QQ",
                type: "video",
                source: "YouTube"
              },
              {
                title: "Eigenvectors for Machine Learning",
                url: "https://towardsdatascience.com/eigenvectors-and-eigenvalues-all-you-need-to-know-df92780f5e0e",
                type: "article",
                source: "Towards Data Science"
              }
            ],
            practice: [
              "Calculate eigenvalues for 5 different matrices",
              "Find corresponding eigenvectors",
              "Implement eigenvalue decomposition using NumPy",
              "Apply PCA to a simple dataset using eigenvectors"
            ],
            timeEstimate: "5-7 hours"
          },
          {
            day: 5,
            title: "Linear Algebra in Neural Networks",
            description: "Applying linear algebra concepts to understand neural networks",
            topics: ["Weight Matrices", "Forward Propagation", "Matrix Chain Rule", "Vectorization"],
            objectives: [
              "Understand how neural networks use matrix operations",
              "Learn vectorization for efficient computation",
              "Implement a simple neural network layer using matrices",
              "Understand the computational efficiency of matrix operations"
            ],
            resources: [
              {
                title: "Deep Learning Specialization - Neural Networks Basics",
                url: "https://www.coursera.org/learn/neural-networks-deep-learning",
                type: "course",
                source: "Coursera"
              },
              {
                title: "Matrix Calculus for Deep Learning",
                url: "https://explained.ai/matrix-calculus/",
                type: "article",
                source: "Explained.ai"
              },
              {
                title: "Neural Networks from Scratch",
                url: "https://www.youtube.com/watch?v=aircAruvnKk",
                type: "video",
                source: "3Blue1Brown"
              }
            ],
            practice: [
              "Implement forward propagation using matrix multiplication",
              "Vectorize a simple neural network computation",
              "Compare vectorized vs loop-based implementations",
              "Build a single-layer perceptron using NumPy"
            ],
            timeEstimate: "6-8 hours"
          },
          {
            day: 6,
            title: "SVD and Matrix Decomposition",
            description: "Singular Value Decomposition and its applications in ML",
            topics: ["SVD", "Matrix Factorization", "Dimensionality Reduction", "Recommender Systems"],
            objectives: [
              "Understand Singular Value Decomposition",
              "Learn about matrix factorization techniques",
              "Apply SVD for dimensionality reduction",
              "Use SVD in recommender systems"
            ],
            resources: [
              {
                title: "StatQuest - SVD and PCA",
                url: "https://www.youtube.com/watch?v=FgakZw6K1QQ",
                type: "video",
                source: "YouTube"
              },
              {
                title: "SVD Tutorial",
                url: "https://web.mit.edu/be.400/www/SVD/Singular_Value_Decomposition.htm",
                type: "tutorial",
                source: "MIT"
              },
              {
                title: "Matrix Factorization for Recommender Systems",
                url: "https://developers.google.com/machine-learning/recommendation/collaborative/matrix",
                type: "article",
                source: "Google Developers"
              }
            ],
            practice: [
              "Perform SVD on sample matrices using NumPy",
              "Implement image compression using SVD",
              "Build a simple movie recommender using matrix factorization",
              "Apply SVD for noise reduction in data"
            ],
            timeEstimate: "5-7 hours"
          },
          {
            day: 7,
            title: "Linear Algebra Project & Review",
            description: "Consolidate linear algebra knowledge with a practical project",
            topics: ["PCA Implementation", "Image Processing", "Data Transformation", "Review"],
            objectives: [
              "Implement PCA from scratch using learned concepts",
              "Apply linear algebra to image compression",
              "Review all linear algebra concepts",
              "Build a portfolio project"
            ],
            resources: [
              {
                title: "PCA from Scratch in Python",
                url: "https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/",
                type: "tutorial",
                source: "ML Mastery"
              },
              {
                title: "Linear Algebra Review for ML",
                url: "https://www.deeplearningbook.org/contents/linear_algebra.html",
                type: "documentation",
                source: "Deep Learning Book"
              },
              {
                title: "Image Compression with Linear Algebra",
                url: "https://towardsdatascience.com/image-compression-using-singular-value-decomposition-svd-43c6ea5bca16",
                type: "article",
                source: "Towards Data Science"
              }
            ],
            practice: [
              "Implement PCA algorithm from scratch",
              "Apply PCA to reduce dimensions of a real dataset",
              "Create an image compression tool using SVD",
              "Document your project and findings"
            ],
            timeEstimate: "6-8 hours"
          }
        ]
      },
      {
        week: 2,
        title: "Calculus Fundamentals",
        days: [
          {
            day: 8,
            title: "Derivatives and Differentiation",
            description: "Understanding derivatives and their role in optimization",
            topics: ["Derivatives", "Chain Rule", "Partial Derivatives", "Gradient"],
            objectives: [
              "Master derivative calculations",
              "Understand the chain rule for composite functions",
              "Learn partial derivatives for multivariable functions",
              "Connect derivatives to neural network backpropagation"
            ],
            resources: [
              {
                title: "3Blue1Brown - Essence of Calculus",
                url: "https://www.youtube.com/watch?v=WUvTyaaNkzM",
                type: "video",
                source: "YouTube"
              },
              {
                title: "Khan Academy - Derivative Calculus",
                url: "https://www.khanacademy.org/math/calculus-1/cs1-derivatives-definition-and-basic-rules",
                type: "course",
                source: "Khan Academy"
              },
              {
                title: "Calculus for Deep Learning",
                url: "https://explained.ai/matrix-calculus/",
                type: "article",
                source: "Explained.ai"
              }
            ],
            practice: [
              "Solve 20 derivative problems of varying difficulty",
              "Practice chain rule with 10 composite functions",
              "Calculate partial derivatives for multivariable functions",
              "Implement automatic differentiation in Python"
            ],
            timeEstimate: "5-7 hours"
          },
          {
            day: 9,
            title: "Gradients and Optimization",
            description: "Understanding gradients, gradient descent, and optimization",
            topics: ["Gradient Vectors", "Gradient Descent", "Learning Rate", "Convergence"],
            objectives: [
              "Understand gradients as vectors of partial derivatives",
              "Learn how gradient descent optimizes functions",
              "Understand the role of learning rate",
              "Implement gradient descent from scratch"
            ],
            resources: [
              {
                title: "StatQuest - Gradient Descent",
                url: "https://www.youtube.com/watch?v=sDv4f4s2SB8",
                type: "video",
                source: "YouTube"
              },
              {
                title: "Gradient Descent Algorithm Explained",
                url: "https://machinelearningmastery.com/gradient-descent-for-machine-learning/",
                type: "article",
                source: "ML Mastery"
              },
              {
                title: "Optimization for Deep Learning",
                url: "https://www.deeplearningbook.org/contents/optimization.html",
                type: "documentation",
                source: "Deep Learning Book"
              }
            ],
            practice: [
              "Implement gradient descent for a simple function",
              "Visualize gradient descent optimization path",
              "Experiment with different learning rates",
              "Apply gradient descent to linear regression"
            ],
            timeEstimate: "5-7 hours"
          },
          {
            day: 10,
            title: "Backpropagation Mathematics",
            description: "Understanding the mathematical foundation of backpropagation",
            topics: ["Chain Rule in Deep Learning", "Computational Graphs", "Backprop Algorithm", "Weight Updates"],
            objectives: [
              "Understand how backpropagation uses the chain rule",
              "Learn to draw and analyze computational graphs",
              "Master the backpropagation algorithm",
              "Calculate gradients for neural network layers"
            ],
            resources: [
              {
                title: "3Blue1Brown - Backpropagation",
                url: "https://www.youtube.com/watch?v=Ilg3gGewQ5U",
                type: "video",
                source: "YouTube"
              },
              {
                title: "CS231n - Backpropagation",
                url: "https://cs231n.github.io/optimization-2/",
                type: "article",
                source: "Stanford CS231n"
              },
              {
                title: "Backpropagation Step by Step",
                url: "https://hmkcode.com/ai/backpropagation-step-by-step/",
                type: "tutorial",
                source: "HMKCODE"
              }
            ],
            practice: [
              "Manually calculate backpropagation for a 2-layer network",
              "Draw computational graphs for various functions",
              "Implement backpropagation from scratch",
              "Verify your implementation against automatic differentiation"
            ],
            timeEstimate: "6-8 hours"
          },
          {
            day: 11,
            title: "Integrals and Probability",
            description: "Understanding integration and its connection to probability",
            topics: ["Integration", "Area Under Curve", "Probability Density Functions", "Expected Value"],
            objectives: [
              "Learn basic integration techniques",
              "Understand definite and indefinite integrals",
              "Connect integration to probability distributions",
              "Calculate expected values using integration"
            ],
            resources: [
              {
                title: "Khan Academy - Integration",
                url: "https://www.khanacademy.org/math/calculus-1/cs1-integrals",
                type: "course",
                source: "Khan Academy"
              },
              {
                title: "Integration in Probability Theory",
                url: "https://seeing-theory.brown.edu/probability-distributions/index.html",
                type: "article",
                source: "Brown University"
              },
              {
                title: "3Blue1Brown - Integration",
                url: "https://www.youtube.com/watch?v=rfG8ce4nNh0",
                type: "video",
                source: "YouTube"
              }
            ],
            practice: [
              "Solve 15 integration problems",
              "Calculate areas under probability density functions",
              "Compute expected values for various distributions",
              "Implement numerical integration in Python"
            ],
            timeEstimate: "4-6 hours"
          },
          {
            day: 12,
            title: "Multivariable Calculus",
            description: "Extending calculus to multiple variables",
            topics: ["Partial Derivatives", "Jacobian Matrix", "Hessian Matrix", "Taylor Series"],
            objectives: [
              "Master partial derivatives for functions of multiple variables",
              "Understand the Jacobian matrix and its applications",
              "Learn about the Hessian and second-order optimization",
              "Apply multivariable calculus to machine learning"
            ],
            resources: [
              {
                title: "Khan Academy - Multivariable Calculus",
                url: "https://www.khanacademy.org/math/multivariable-calculus",
                type: "course",
                source: "Khan Academy"
              },
              {
                title: "Jacobian and Hessian Matrices",
                url: "https://www.youtube.com/watch?v=bohL918kXQk",
                type: "video",
                source: "YouTube"
              },
              {
                title: "Second-Order Optimization Methods",
                url: "https://towardsdatascience.com/second-order-optimization-for-neural-networks-42e8e4454f7f",
                type: "article",
                source: "Towards Data Science"
              }
            ],
            practice: [
              "Calculate Jacobian matrices for vector-valued functions",
              "Compute Hessian matrices for scalar functions",
              "Implement Newton's method for optimization",
              "Apply second-order methods to a simple problem"
            ],
            timeEstimate: "5-7 hours"
          },
          {
            day: 13,
            title: "Optimization Algorithms",
            description: "Advanced optimization techniques for deep learning",
            topics: ["SGD", "Momentum", "Adam", "RMSprop", "Learning Rate Schedules"],
            objectives: [
              "Understand stochastic gradient descent and its variants",
              "Learn momentum-based optimization",
              "Master Adam and adaptive learning rate methods",
              "Implement various optimizers from scratch"
            ],
            resources: [
              {
                title: "An Overview of Gradient Descent Optimization Algorithms",
                url: "https://arxiv.org/abs/1609.04747",
                type: "article",
                source: "arXiv"
              },
              {
                title: "Optimizers Explained",
                url: "https://www.youtube.com/watch?v=mdKjMPmcWjY",
                type: "video",
                source: "YouTube"
              },
              {
                title: "Deep Learning Optimizers",
                url: "https://pytorch.org/docs/stable/optim.html",
                type: "documentation",
                source: "PyTorch"
              }
            ],
            practice: [
              "Implement SGD with momentum from scratch",
              "Implement Adam optimizer",
              "Compare performance of different optimizers",
              "Tune hyperparameters for various optimizers"
            ],
            timeEstimate: "6-8 hours"
          },
          {
            day: 14,
            title: "Calculus Project & Review",
            description: "Apply calculus concepts to build a neural network optimizer",
            topics: ["Custom Optimizer", "Loss Landscape Visualization", "Convergence Analysis"],
            objectives: [
              "Build a complete neural network training loop",
              "Implement multiple optimizers and compare them",
              "Visualize loss landscapes and optimization paths",
              "Analyze convergence behavior"
            ],
            resources: [
              {
                title: "Visualizing Loss Landscapes",
                url: "https://losslandscape.com/",
                type: "article",
                source: "Research"
              },
              {
                title: "Neural Network Optimization",
                url: "https://www.deeplearningbook.org/contents/optimization.html",
                type: "documentation",
                source: "Deep Learning Book"
              },
              {
                title: "Building Optimizers from Scratch",
                url: "https://pytorch.org/tutorials/beginner/examples_nn/polynomial_custom_function.html",
                type: "tutorial",
                source: "PyTorch"
              }
            ],
            practice: [
              "Build a neural network trainer with multiple optimizers",
              "Visualize and compare optimization trajectories",
              "Create loss landscape visualizations",
              "Write a comprehensive report on your findings"
            ],
            timeEstimate: "7-9 hours"
          }
        ]
      },
      {
        week: 3,
        title: "Probability and Statistics",
        days: [
          {
            day: 15,
            title: "Probability Fundamentals",
            description: "Basic concepts of probability theory",
            topics: ["Sample Spaces", "Events", "Probability Rules", "Conditional Probability"],
            objectives: [
              "Understand basic probability concepts and notation",
              "Learn probability rules and axioms",
              "Master conditional probability and Bayes' theorem",
              "Apply probability to machine learning problems"
            ],
            resources: [
              {
                title: "Khan Academy - Probability",
                url: "https://www.khanacademy.org/math/statistics-probability/probability-library",
                type: "course",
                source: "Khan Academy"
              },
              {
                title: "Seeing Theory - Probability Visualization",
                url: "https://seeing-theory.brown.edu/basic-probability/index.html",
                type: "article",
                source: "Brown University"
              },
              {
                title: "StatQuest - Probability",
                url: "https://www.youtube.com/watch?v=uzkc-qNVoOk",
                type: "video",
                source: "YouTube"
              }
            ],
            practice: [
              "Solve 20 probability problems",
              "Apply Bayes' theorem to 5 real-world scenarios",
              "Implement probability calculations in Python",
              "Solve conditional probability problems"
            ],
            timeEstimate: "4-6 hours"
          },
          {
            day: 16,
            title: "Probability Distributions",
            description: "Common probability distributions and their applications",
            topics: ["Bernoulli", "Binomial", "Normal Distribution", "Uniform Distribution"],
            objectives: [
              "Understand discrete and continuous distributions",
              "Learn properties of normal distribution",
              "Master the central limit theorem",
              "Apply distributions to model real data"
            ],
            resources: [
              {
                title: "StatQuest - Probability Distributions",
                url: "https://www.youtube.com/watch?v=YXLVjCKVP7U",
                type: "video",
                source: "YouTube"
              },
              {
                title: "Probability Distributions Guide",
                url: "https://seeing-theory.brown.edu/probability-distributions/index.html",
                type: "article",
                source: "Brown University"
              },
              {
                title: "SciPy Statistics Documentation",
                url: "https://docs.scipy.org/doc/scipy/reference/stats.html",
                type: "documentation",
                source: "SciPy"
              }
            ],
            practice: [
              "Plot various probability distributions using Python",
              "Calculate probabilities from different distributions",
              "Verify central limit theorem with simulations",
              "Model real-world data with appropriate distributions"
            ],
            timeEstimate: "5-7 hours"
          },
          {
            day: 17,
            title: "Bayes' Theorem and Bayesian Thinking",
            description: "Deep dive into Bayesian inference and its ML applications",
            topics: ["Bayes' Theorem", "Prior & Posterior", "Bayesian Inference", "Naive Bayes"],
            objectives: [
              "Master Bayes' theorem and its applications",
              "Understand prior, likelihood, and posterior",
              "Learn Bayesian updating",
              "Implement Naive Bayes classifier"
            ],
            resources: [
              {
                title: "3Blue1Brown - Bayes Theorem",
                url: "https://www.youtube.com/watch?v=HZGCoVF3YvM",
                type: "video",
                source: "YouTube"
              },
              {
                title: "Bayesian Statistics Explained",
                url: "https://towardsdatascience.com/bayesian-statistics-explained-to-beginners-in-simple-english-f55fb02e74ba",
                type: "article",
                source: "Towards Data Science"
              },
              {
                title: "Naive Bayes Classifier Tutorial",
                url: "https://scikit-learn.org/stable/modules/naive_bayes.html",
                type: "documentation",
                source: "Scikit-learn"
              }
            ],
            practice: [
              "Solve 10 Bayes' theorem problems",
              "Implement Naive Bayes from scratch",
              "Apply Naive Bayes to text classification",
              "Compare with sklearn implementation"
            ],
            timeEstimate: "5-7 hours"
          },
          {
            day: 18,
            title: "Statistical Inference",
            description: "Hypothesis testing, confidence intervals, and p-values",
            topics: ["Hypothesis Testing", "P-values", "Confidence Intervals", "Statistical Significance"],
            objectives: [
              "Understand hypothesis testing framework",
              "Learn to interpret p-values correctly",
              "Calculate and interpret confidence intervals",
              "Apply statistical tests to real data"
            ],
            resources: [
              {
                title: "StatQuest - Hypothesis Testing",
                url: "https://www.youtube.com/watch?v=0oc49DyA3hU",
                type: "video",
                source: "YouTube"
              },
              {
                title: "Statistical Inference Guide",
                url: "https://seeing-theory.brown.edu/frequentist-inference/index.html",
                type: "article",
                source: "Brown University"
              },
              {
                title: "P-values Explained",
                url: "https://www.nature.com/articles/506150a",
                type: "article",
                source: "Nature"
              }
            ],
            practice: [
              "Perform t-tests on sample datasets",
              "Calculate confidence intervals for various parameters",
              "Interpret p-values from statistical tests",
              "Conduct A/B testing simulation"
            ],
            timeEstimate: "4-6 hours"
          },
          {
            day: 19,
            title: "Maximum Likelihood Estimation",
            description: "Understanding MLE and its role in machine learning",
            topics: ["Likelihood Function", "MLE", "Log-Likelihood", "MLE in ML Models"],
            objectives: [
              "Understand the concept of likelihood",
              "Learn maximum likelihood estimation",
              "Master log-likelihood optimization",
              "Apply MLE to fit distributions and models"
            ],
            resources: [
              {
                title: "StatQuest - Maximum Likelihood",
                url: "https://www.youtube.com/watch?v=XepXtl9YKwc",
                type: "video",
                source: "YouTube"
              },
              {
                title: "MLE for Machine Learning",
                url: "https://towardsdatascience.com/probability-concepts-explained-maximum-likelihood-estimation-c7b4342fdbb1",
                type: "article",
                source: "Towards Data Science"
              },
              {
                title: "MLE Tutorial",
                url: "https://machinelearningmastery.com/what-is-maximum-likelihood-estimation-in-machine-learning/",
                type: "tutorial",
                source: "ML Mastery"
              }
            ],
            practice: [
              "Derive MLE for normal distribution parameters",
              "Implement MLE for various distributions",
              "Apply MLE to logistic regression",
              "Compare MLE with other estimation methods"
            ],
            timeEstimate: "5-7 hours"
          },
          {
            day: 20,
            title: "Information Theory Basics",
            description: "Entropy, cross-entropy, and KL divergence",
            topics: ["Entropy", "Cross-Entropy", "KL Divergence", "Mutual Information"],
            objectives: [
              "Understand information entropy",
              "Learn cross-entropy loss function",
              "Master KL divergence for comparing distributions",
              "Apply information theory to ML problems"
            ],
            resources: [
              {
                title: "Visual Information Theory",
                url: "https://colah.github.io/posts/2015-09-Visual-Information/",
                type: "article",
                source: "Colah's Blog"
              },
              {
                title: "Entropy and Information Gain",
                url: "https://www.youtube.com/watch?v=IPkRVpXtbdY",
                type: "video",
                source: "YouTube"
              },
              {
                title: "Cross-Entropy Loss Explained",
                url: "https://machinelearningmastery.com/cross-entropy-for-machine-learning/",
                type: "article",
                source: "ML Mastery"
              }
            ],
            practice: [
              "Calculate entropy for various distributions",
              "Implement cross-entropy loss function",
              "Calculate KL divergence between distributions",
              "Apply entropy for feature selection in decision trees"
            ],
            timeEstimate: "4-6 hours"
          },
          {
            day: 21,
            title: "Statistics Project & Review",
            description: "Statistical analysis and A/B testing project",
            topics: ["A/B Testing", "Statistical Analysis", "Bayesian vs Frequentist", "Review"],
            objectives: [
              "Conduct a complete A/B testing analysis",
              "Apply statistical inference to real data",
              "Compare Bayesian and Frequentist approaches",
              "Build a statistical analysis toolkit"
            ],
            resources: [
              {
                title: "A/B Testing Guide",
                url: "https://www.optimizely.com/optimization-glossary/ab-testing/",
                type: "article",
                source: "Optimizely"
              },
              {
                title: "Statistical Analysis in Python",
                url: "https://scipy-lectures.org/packages/statistics/index.html",
                type: "tutorial",
                source: "SciPy Lectures"
              },
              {
                title: "Bayesian vs Frequentist",
                url: "https://www.youtube.com/watch?v=eDMGDhyDxuY",
                type: "video",
                source: "YouTube"
              }
            ],
            practice: [
              "Design and simulate an A/B test",
              "Perform statistical analysis on real dataset",
              "Compare Bayesian and Frequentist results",
              "Create a comprehensive analysis report"
            ],
            timeEstimate: "6-8 hours"
          }
        ]
      },
      {
        week: 4,
        title: "Information Theory & Discrete Math",
        days: [
          {
            day: 22,
            title: "Graph Theory Fundamentals",
            description: "Graphs, trees, and their applications in AI",
            topics: ["Graph Representation", "Trees", "Graph Traversal", "Shortest Paths"],
            objectives: [
              "Understand graph data structures",
              "Learn BFS and DFS algorithms",
              "Master tree structures and traversals",
              "Apply graphs to AI problems (search, neural networks)"
            ],
            resources: [
              {
                title: "Graph Theory Tutorial",
                url: "https://www.youtube.com/watch?v=LFKZLXVO-Dg",
                type: "video",
                source: "YouTube"
              },
              {
                title: "Graphs in Machine Learning",
                url: "https://towardsdatascience.com/graph-theory-and-deep-learning-know-hows-6556b0e9891b",
                type: "article",
                source: "Towards Data Science"
              },
              {
                title: "NetworkX Tutorial",
                url: "https://networkx.org/documentation/stable/tutorial.html",
                type: "documentation",
                source: "NetworkX"
              }
            ],
            practice: [
              "Implement graph representations (adjacency matrix/list)",
              "Code BFS and DFS from scratch",
              "Solve shortest path problems",
              "Apply graphs to represent neural network architectures"
            ],
            timeEstimate: "5-7 hours"
          },
          {
            day: 23,
            title: "Combinatorics and Counting",
            description: "Permutations, combinations, and counting principles",
            topics: ["Permutations", "Combinations", "Counting Principles", "Binomial Theorem"],
            objectives: [
              "Master permutation and combination formulas",
              "Learn counting principles",
              "Apply combinatorics to probability",
              "Understand computational complexity analysis"
            ],
            resources: [
              {
                title: "Khan Academy - Combinatorics",
                url: "https://www.khanacademy.org/math/precalculus/x9e81a4f98389efdf:prob-comb",
                type: "course",
                source: "Khan Academy"
              },
              {
                title: "Combinatorics in Computer Science",
                url: "https://www.youtube.com/watch?v=p8vIcmr_Pqo",
                type: "video",
                source: "YouTube"
              },
              {
                title: "Counting and Probability",
                url: "https://brilliant.org/wiki/counting-principles/",
                type: "article",
                source: "Brilliant"
              }
            ],
            practice: [
              "Solve 15 permutation and combination problems",
              "Apply counting to calculate probabilities",
              "Implement combinations generator in Python",
              "Analyze algorithm complexity using combinatorics"
            ],
            timeEstimate: "4-5 hours"
          },
          {
            day: 24,
            title: "Logic and Boolean Algebra",
            description: "Propositional logic and its applications",
            topics: ["Propositional Logic", "Truth Tables", "Boolean Algebra", "Logic Gates"],
            objectives: [
              "Understand logical operators and propositions",
              "Create and analyze truth tables",
              "Learn Boolean algebra laws",
              "Connect logic to neural network activations"
            ],
            resources: [
              {
                title: "Boolean Algebra Tutorial",
                url: "https://www.youtube.com/watch?v=gI-qXk7XojA",
                type: "video",
                source: "YouTube"
              },
              {
                title: "Logic for Computer Science",
                url: "https://www.cs.utexas.edu/~isil/cs311h/lecture-propositional-logic.pdf",
                type: "article",
                source: "UT Austin"
              },
              {
                title: "Boolean Logic in Programming",
                url: "https://realpython.com/python-boolean/",
                type: "tutorial",
                source: "Real Python"
              }
            ],
            practice: [
              "Create truth tables for complex propositions",
              "Simplify Boolean expressions",
              "Implement logic gates in Python",
              "Build a simple logic circuit simulator"
            ],
            timeEstimate: "3-5 hours"
          },
          {
            day: 25,
            title: "Set Theory and Relations",
            description: "Set operations and their role in data science",
            topics: ["Sets", "Set Operations", "Relations", "Functions"],
            objectives: [
              "Master set theory notation and operations",
              "Understand relations and functions formally",
              "Apply set theory to data manipulation",
              "Use sets in algorithm design"
            ],
            resources: [
              {
                title: "Set Theory Basics",
                url: "https://www.youtube.com/watch?v=tyDKR4FG3Yw",
                type: "video",
                source: "YouTube"
              },
              {
                title: "Set Theory for Data Science",
                url: "https://towardsdatascience.com/set-theory-for-data-science-afb23abab237",
                type: "article",
                source: "Towards Data Science"
              },
              {
                title: "Python Sets Tutorial",
                url: "https://realpython.com/python-sets/",
                type: "tutorial",
                source: "Real Python"
              }
            ],
            practice: [
              "Solve set theory problems (unions, intersections)",
              "Implement set operations efficiently",
              "Use sets for data deduplication",
              "Apply set theory to feature engineering"
            ],
            timeEstimate: "3-5 hours"
          },
          {
            day: 26,
            title: "Complexity Theory Basics",
            description: "Big O notation and algorithm complexity",
            topics: ["Big O Notation", "Time Complexity", "Space Complexity", "Algorithm Analysis"],
            objectives: [
              "Understand Big O, Omega, and Theta notations",
              "Analyze time and space complexity",
              "Compare algorithm efficiency",
              "Optimize code for better performance"
            ],
            resources: [
              {
                title: "Big O Notation Explained",
                url: "https://www.youtube.com/watch?v=__vX2sjlpXU",
                type: "video",
                source: "YouTube"
              },
              {
                title: "Complexity Analysis Guide",
                url: "https://www.bigocheatsheet.com/",
                type: "article",
                source: "Big-O Cheat Sheet"
              },
              {
                title: "Algorithm Complexity",
                url: "https://www.geeksforgeeks.org/understanding-time-complexity-simple-examples/",
                type: "tutorial",
                source: "GeeksforGeeks"
              }
            ],
            practice: [
              "Analyze complexity of 10 different algorithms",
              "Optimize code to reduce time complexity",
              "Compare sorting algorithm complexities",
              "Profile Python code for performance"
            ],
            timeEstimate: "4-6 hours"
          },
          {
            day: 27,
            title: "Numerical Methods",
            description: "Numerical computation and approximation techniques",
            topics: ["Numerical Stability", "Floating Point", "Approximation Methods", "Error Analysis"],
            objectives: [
              "Understand floating-point arithmetic limitations",
              "Learn numerical stability concepts",
              "Master approximation methods",
              "Handle numerical errors in ML implementations"
            ],
            resources: [
              {
                title: "Numerical Computing in Python",
                url: "https://www.youtube.com/watch?v=E43-CfukEgs",
                type: "video",
                source: "YouTube"
              },
              {
                title: "Floating Point Arithmetic",
                url: "https://docs.python.org/3/tutorial/floatingpoint.html",
                type: "documentation",
                source: "Python Docs"
              },
              {
                title: "Numerical Stability in Deep Learning",
                url: "https://www.deeplearningbook.org/contents/numerical.html",
                type: "article",
                source: "Deep Learning Book"
              }
            ],
            practice: [
              "Explore floating-point precision issues",
              "Implement numerically stable algorithms",
              "Compare different numerical methods",
              "Apply log-sum-exp trick for stability"
            ],
            timeEstimate: "4-6 hours"
          },
          {
            day: 28,
            title: "Month 1 Capstone Project",
            description: "Comprehensive project applying all mathematical concepts",
            topics: ["Full ML Pipeline", "Mathematical Foundations", "Documentation"],
            objectives: [
              "Build a complete ML model from scratch",
              "Apply linear algebra, calculus, and statistics",
              "Implement optimizers and loss functions",
              "Create comprehensive documentation"
            ],
            resources: [
              {
                title: "Machine Learning from Scratch",
                url: "https://github.com/eriklindernoren/ML-From-Scratch",
                type: "tutorial",
                source: "GitHub"
              },
              {
                title: "Build Neural Network from Scratch",
                url: "https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65",
                type: "article",
                source: "Towards Data Science"
              },
              {
                title: "Deep Learning Book",
                url: "https://www.deeplearningbook.org/",
                type: "documentation",
                source: "Deep Learning Book"
              }
            ],
            practice: [
              "Build a neural network from scratch using only NumPy",
              "Implement backpropagation manually",
              "Train on a real dataset (MNIST or similar)",
              "Write detailed mathematical documentation",
              "Create visualizations of learning process"
            ],
            timeEstimate: "8-10 hours"
          }
        ]
      }
    ]
  },
  {
    month: 2,
    title: "Python Programming & Data Structures",
    weeks: [
      {
        week: 1,
        title: "Python Fundamentals & NumPy",
        days: [
          {
            day: 29,
            title: "Python Deep Dive",
            description: "Advanced Python concepts for AI development",
            topics: ["OOP", "Decorators", "Generators", "List Comprehensions"],
            objectives: [
              "Master object-oriented programming in Python",
              "Learn decorators and their applications",
              "Understand generators for memory efficiency",
              "Write Pythonic code using comprehensions"
            ],
            resources: [
              {
                title: "Real Python - OOP Tutorial",
                url: "https://realpython.com/python3-object-oriented-programming/",
                type: "tutorial",
                source: "Real Python"
              },
              {
                title: "Python Decorators Explained",
                url: "https://www.youtube.com/watch?v=FsAPt_9Bf3U",
                type: "video",
                source: "YouTube"
              },
              {
                title: "Advanced Python Features",
                url: "https://docs.python.org/3/tutorial/",
                type: "documentation",
                source: "Python Docs"
              }
            ],
            practice: [
              "Build a class hierarchy for neural network layers",
              "Create custom decorators for timing and logging",
              "Implement data pipeline using generators",
              "Refactor code using list comprehensions"
            ],
            timeEstimate: "5-7 hours"
          },
          {
            day: 30,
            title: "NumPy Mastery",
            description: "Advanced NumPy for efficient numerical computing",
            topics: ["Broadcasting", "Vectorization", "Advanced Indexing", "Linear Algebra in NumPy"],
            objectives: [
              "Master NumPy broadcasting rules",
              "Write vectorized code for performance",
              "Use advanced indexing techniques",
              "Perform efficient linear algebra operations"
            ],
            resources: [
              {
                title: "NumPy Documentation",
                url: "https://numpy.org/doc/stable/user/basics.html",
                type: "documentation",
                source: "NumPy"
              },
              {
                title: "NumPy Tutorial - FreeCodeCamp",
                url: "https://www.youtube.com/watch?v=QUT1VHiLmmI",
                type: "video",
                source: "YouTube"
              },
              {
                title: "From Python to NumPy",
                url: "https://www.labri.fr/perso/nrougier/from-python-to-numpy/",
                type: "article",
                source: "Nicolas Rougier"
              }
            ],
            practice: [
              "Implement matrix operations using broadcasting",
              "Optimize loops by vectorization (10x speedup)",
              "Use fancy indexing for data manipulation",
              "Build efficient neural network operations"
            ],
            timeEstimate: "5-7 hours"
          }
        ]
      },
      {
        week: 2,
        title: "Pandas & Data Manipulation",
        days: [
          {
            day: 31,
            title: "Pandas Fundamentals",
            description: "Master data manipulation with Pandas",
            topics: ["DataFrames", "Series", "Indexing", "Data Cleaning"],
            objectives: [
              "Understand DataFrame and Series structures",
              "Master data indexing and selection",
              "Learn data cleaning techniques",
              "Handle missing data effectively"
            ],
            resources: [
              {
                title: "Pandas Documentation",
                url: "https://pandas.pydata.org/docs/user_guide/index.html",
                type: "documentation",
                source: "Pandas"
              },
              {
                title: "Pandas Tutorial - Corey Schafer",
                url: "https://www.youtube.com/watch?v=ZyhVh-qRZPA",
                type: "video",
                source: "YouTube"
              },
              {
                title: "10 Minutes to Pandas",
                url: "https://pandas.pydata.org/docs/user_guide/10min.html",
                type: "tutorial",
                source: "Pandas"
              }
            ],
            practice: [
              "Load and explore real datasets",
              "Clean data with missing values",
              "Perform complex data selections",
              "Transform data for ML pipelines"
            ],
            timeEstimate: "5-7 hours"
          }
        ]
      },
      {
        week: 3,
        title: "Data Structures & Algorithms",
        days: [
          {
            day: 32,
            title: "Arrays and Linked Lists",
            description: "Fundamental data structures for AI",
            topics: ["Arrays", "Dynamic Arrays", "Linked Lists", "Implementation"],
            objectives: [
              "Understand array operations and complexity",
              "Implement linked lists from scratch",
              "Compare array vs linked list performance",
              "Apply to AI data structures"
            ],
            resources: [
              {
                title: "Data Structures - CS50",
                url: "https://www.youtube.com/watch?v=4IrUAqYKjIA",
                type: "video",
                source: "YouTube"
              },
              {
                title: "Python Data Structures",
                url: "https://realpython.com/python-data-structures/",
                type: "tutorial",
                source: "Real Python"
              }
            ],
            practice: [
              "Implement array operations from scratch",
              "Build singly and doubly linked lists",
              "Solve 10 array manipulation problems",
              "Optimize array-based algorithms"
            ],
            timeEstimate: "5-6 hours"
          }
        ]
      },
      {
        week: 4,
        title: "Advanced Python & ML Libraries",
        days: [
          {
            day: 33,
            title: "Month 2 Project - Data Pipeline",
            description: "Build an end-to-end data processing pipeline",
            topics: ["ETL Pipeline", "Data Validation", "Automation", "Documentation"],
            objectives: [
              "Build a complete data pipeline",
              "Implement data validation and quality checks",
              "Automate data processing workflow",
              "Create comprehensive documentation"
            ],
            resources: [
              {
                title: "Building Data Pipelines",
                url: "https://towardsdatascience.com/a-beginners-guide-to-data-engineering-part-i-4227c5c457d7",
                type: "article",
                source: "Towards Data Science"
              }
            ],
            practice: [
              "Design and implement an ETL pipeline",
              "Add data validation and error handling",
              "Automate pipeline execution",
              "Write comprehensive tests",
              "Create documentation and visualizations"
            ],
            timeEstimate: "8-10 hours"
          }
        ]
      }
    ]
  },
  {
    month: 3,
    title: "Machine Learning Fundamentals",
    weeks: [
      {
        week: 1,
        title: "Introduction to Machine Learning",
        days: [
          {
            day: 34,
            title: "ML Overview and Types",
            description: "Introduction to machine learning concepts",
            topics: ["Supervised Learning", "Unsupervised Learning", "Reinforcement Learning", "ML Workflow"],
            objectives: [
              "Understand different types of machine learning",
              "Learn the ML development workflow",
              "Identify appropriate ML approaches for problems",
              "Understand bias-variance tradeoff"
            ],
            resources: [
              {
                title: "Machine Learning Crash Course - Google",
                url: "https://developers.google.com/machine-learning/crash-course",
                type: "course",
                source: "Google"
              },
              {
                title: "StatQuest - ML Fundamentals",
                url: "https://www.youtube.com/watch?v=Gv9_4yMHFhI",
                type: "video",
                source: "YouTube"
              }
            ],
            practice: [
              "Categorize 10 real-world problems by ML type",
              "Design ML workflows for different scenarios",
              "Analyze bias-variance in simple models",
              "Set up scikit-learn environment"
            ],
            timeEstimate: "5-6 hours"
          },
          {
            day: 35,
            title: "Linear Regression",
            description: "Understanding and implementing linear regression",
            topics: ["Simple Linear Regression", "Multiple Regression", "Gradient Descent", "Evaluation Metrics"],
            objectives: [
              "Implement linear regression from scratch",
              "Use gradient descent for optimization",
              "Evaluate models using MSE, RMSE, R²",
              "Apply regularization techniques"
            ],
            resources: [
              {
                title: "StatQuest - Linear Regression",
                url: "https://www.youtube.com/watch?v=nk2CQITm_eo",
                type: "video",
                source: "YouTube"
              },
              {
                title: "Scikit-learn Linear Models",
                url: "https://scikit-learn.org/stable/modules/linear_model.html",
                type: "documentation",
                source: "Scikit-learn"
              }
            ],
            practice: [
              "Implement linear regression from scratch",
              "Train on real dataset (housing prices)",
              "Compare with scikit-learn implementation",
              "Visualize predictions and residuals"
            ],
            timeEstimate: "6-7 hours"
          }
        ]
      },
      {
        week: 2,
        title: "Classical ML Algorithms",
        days: [
          {
            day: 36,
            title: "Decision Trees",
            description: "Understanding tree-based models",
            topics: ["Decision Trees", "Information Gain", "Pruning", "CART Algorithm"],
            objectives: [
              "Understand how decision trees make decisions",
              "Learn splitting criteria (Gini, entropy)",
              "Implement decision tree from scratch",
              "Apply to classification and regression"
            ],
            resources: [
              {
                title: "StatQuest - Decision Trees",
                url: "https://www.youtube.com/watch?v=_L39rN6gz7Y",
                type: "video",
                source: "YouTube"
              },
              {
                title: "Decision Trees in Scikit-learn",
                url: "https://scikit-learn.org/stable/modules/tree.html",
                type: "documentation",
                source: "Scikit-learn"
              }
            ],
            practice: [
              "Implement decision tree classifier",
              "Visualize decision boundaries",
              "Apply pruning to prevent overfitting",
              "Build regression tree"
            ],
            timeEstimate: "6-7 hours"
          },
          {
            day: 37,
            title: "Random Forests",
            description: "Ensemble learning with random forests",
            topics: ["Ensemble Methods", "Bagging", "Random Forests", "Feature Importance"],
            objectives: [
              "Understand ensemble learning principles",
              "Learn bagging and random subspaces",
              "Implement random forest classifier",
              "Extract feature importance"
            ],
            resources: [
              {
                title: "StatQuest - Random Forests",
                url: "https://www.youtube.com/watch?v=J4Wdy0Wc_xQ",
                type: "video",
                source: "YouTube"
              },
              {
                title: "Random Forest Guide",
                url: "https://towardsdatascience.com/understanding-random-forest-58381e0602d2",
                type: "article",
                source: "Towards Data Science"
              }
            ],
            practice: [
              "Build random forest from scratch",
              "Compare with single decision tree",
              "Analyze feature importance",
              "Apply to Kaggle competition"
            ],
            timeEstimate: "6-7 hours"
          }
        ]
      },
      {
        week: 3,
        title: "Unsupervised Learning",
        days: [
          {
            day: 38,
            title: "K-Means Clustering",
            description: "Introduction to clustering algorithms",
            topics: ["K-Means", "Elbow Method", "Cluster Evaluation", "Applications"],
            objectives: [
              "Understand clustering concepts",
              "Implement K-means from scratch",
              "Choose optimal number of clusters",
              "Apply clustering to real problems"
            ],
            resources: [
              {
                title: "StatQuest - K-means Clustering",
                url: "https://www.youtube.com/watch?v=4b5d3muPQmA",
                type: "video",
                source: "YouTube"
              },
              {
                title: "Clustering in Scikit-learn",
                url: "https://scikit-learn.org/stable/modules/clustering.html",
                type: "documentation",
                source: "Scikit-learn"
              }
            ],
            practice: [
              "Implement K-means clustering",
              "Use elbow method to find K",
              "Visualize clusters in 2D/3D",
              "Apply to customer segmentation"
            ],
            timeEstimate: "5-6 hours"
          },
          {
            day: 39,
            title: "Dimensionality Reduction",
            description: "PCA and feature reduction techniques",
            topics: ["PCA", "t-SNE", "Feature Selection", "Curse of Dimensionality"],
            objectives: [
              "Understand dimensionality reduction need",
              "Implement PCA from scratch",
              "Use t-SNE for visualization",
              "Apply to high-dimensional data"
            ],
            resources: [
              {
                title: "StatQuest - PCA",
                url: "https://www.youtube.com/watch?v=FgakZw6K1QQ",
                type: "video",
                source: "YouTube"
              },
              {
                title: "PCA in Scikit-learn",
                url: "https://scikit-learn.org/stable/modules/decomposition.html#pca",
                type: "documentation",
                source: "Scikit-learn"
              }
            ],
            practice: [
              "Implement PCA for dimensionality reduction",
              "Visualize high-dimensional data with t-SNE",
              "Compare reconstruction error",
              "Apply to image compression"
            ],
            timeEstimate: "5-6 hours"
          }
        ]
      },
      {
        week: 4,
        title: "Model Selection & Deployment",
        days: [
          {
            day: 40,
            title: "Month 3 Project - ML Competition",
            description: "Complete Kaggle-style machine learning competition",
            topics: ["Feature Engineering", "Model Selection", "Hyperparameter Tuning", "Ensemble Methods"],
            objectives: [
              "Build complete ML pipeline",
              "Engineer effective features",
              "Select and tune models",
              "Create final predictions"
            ],
            resources: [
              {
                title: "Kaggle Learn",
                url: "https://www.kaggle.com/learn",
                type: "course",
                source: "Kaggle"
              },
              {
                title: "Feature Engineering Guide",
                url: "https://www.featuretools.com/",
                type: "article",
                source: "Feature Tools"
              }
            ],
            practice: [
              "Complete a Kaggle competition",
              "Engineer domain-specific features",
              "Build ensemble model",
              "Write competition report"
            ],
            timeEstimate: "10-12 hours"
          }
        ]
      }
    ]
  },
  {
    month: 4,
    title: "Deep Learning & Neural Networks",
    weeks: [
      {
        week: 1,
        title: "Neural Network Fundamentals",
        days: [
          {
            day: 41,
            title: "Perceptrons and Multilayer Networks",
            description: "Building blocks of neural networks",
            topics: ["Perceptron", "Multilayer Perceptron", "Activation Functions", "Forward Propagation"],
            objectives: [
              "Understand perceptron algorithm",
              "Build multilayer neural networks",
              "Learn different activation functions",
              "Implement forward propagation"
            ],
            resources: [
              {
                title: "3Blue1Brown - Neural Networks",
                url: "https://www.youtube.com/watch?v=aircAruvnKk",
                type: "video",
                source: "YouTube"
              },
              {
                title: "Neural Networks from Scratch",
                url: "https://nnfs.io/",
                type: "course",
                source: "NNFS"
              }
            ],
            practice: [
              "Implement perceptron from scratch",
              "Build multi-layer network",
              "Test different activation functions",
              "Visualize network activations"
            ],
            timeEstimate: "7-8 hours"
          },
          {
            day: 42,
            title: "Introduction to PyTorch",
            description: "Deep learning with PyTorch framework",
            topics: ["Tensors", "Autograd", "Neural Network Modules", "Training Loop"],
            objectives: [
              "Master PyTorch tensor operations",
              "Understand automatic differentiation",
              "Build networks using nn.Module",
              "Implement training loops"
            ],
            resources: [
              {
                title: "PyTorch Tutorials",
                url: "https://pytorch.org/tutorials/",
                type: "documentation",
                source: "PyTorch"
              },
              {
                title: "PyTorch for Deep Learning",
                url: "https://www.youtube.com/watch?v=V_xro1bcAuA",
                type: "video",
                source: "YouTube"
              }
            ],
            practice: [
              "Convert NumPy code to PyTorch",
              "Build neural network with nn.Module",
              "Implement training loop",
              "Train on MNIST dataset"
            ],
            timeEstimate: "7-8 hours"
          }
        ]
      },
      {
        week: 2,
        title: "Convolutional Neural Networks",
        days: [
          {
            day: 43,
            title: "CNN Architecture",
            description: "Understanding convolutional networks",
            topics: ["Convolution Layers", "Pooling", "CNN Architecture", "Image Classification"],
            objectives: [
              "Understand convolution operation",
              "Learn pooling and stride",
              "Build CNN architectures",
              "Apply to image classification"
            ],
            resources: [
              {
                title: "Stanford CS231n - CNNs",
                url: "https://cs231n.github.io/convolutional-networks/",
                type: "article",
                source: "Stanford"
              },
              {
                title: "CNN Explainer",
                url: "https://poloclub.github.io/cnn-explainer/",
                type: "article",
                source: "Polo Club"
              }
            ],
            practice: [
              "Implement convolution from scratch",
              "Build CNN in PyTorch",
              "Train on CIFAR-10",
              "Visualize learned filters"
            ],
            timeEstimate: "8-9 hours"
          },
          {
            day: 44,
            title: "Transfer Learning",
            description: "Using pre-trained models",
            topics: ["Pre-trained Models", "Fine-tuning", "Feature Extraction", "Domain Adaptation"],
            objectives: [
              "Understand transfer learning concepts",
              "Use pre-trained models (ResNet, VGG)",
              "Fine-tune for custom datasets",
              "Extract features for downstream tasks"
            ],
            resources: [
              {
                title: "Transfer Learning Guide",
                url: "https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html",
                type: "tutorial",
                source: "PyTorch"
              },
              {
                title: "Fine-tuning Models",
                url: "https://www.youtube.com/watch?v=5T-iXNNiwIs",
                type: "video",
                source: "YouTube"
              }
            ],
            practice: [
              "Load pre-trained ResNet",
              "Fine-tune on custom dataset",
              "Compare from-scratch vs transfer learning",
              "Build image classifier"
            ],
            timeEstimate: "7-8 hours"
          }
        ]
      },
      {
        week: 3,
        title: "Recurrent Neural Networks",
        days: [
          {
            day: 45,
            title: "RNNs and LSTMs",
            description: "Sequential data processing",
            topics: ["RNN", "LSTM", "GRU", "Sequence Modeling"],
            objectives: [
              "Understand recurrent architectures",
              "Learn LSTM and GRU cells",
              "Handle variable-length sequences",
              "Apply to time series and NLP"
            ],
            resources: [
              {
                title: "Understanding LSTMs",
                url: "https://colah.github.io/posts/2015-08-Understanding-LSTMs/",
                type: "article",
                source: "Colah's Blog"
              },
              {
                title: "RNN Tutorial",
                url: "https://www.youtube.com/watch?v=LHXXI4-IEns",
                type: "video",
                source: "YouTube"
              }
            ],
            practice: [
              "Implement RNN from scratch",
              "Build LSTM in PyTorch",
              "Train on time series data",
              "Generate text with RNN"
            ],
            timeEstimate: "8-9 hours"
          },
          {
            day: 46,
            title: "Attention Mechanisms",
            description: "Introduction to attention in neural networks",
            topics: ["Attention", "Self-Attention", "Multi-Head Attention", "Transformers Intro"],
            objectives: [
              "Understand attention mechanism",
              "Implement self-attention",
              "Learn multi-head attention",
              "Introduction to Transformer architecture"
            ],
            resources: [
              {
                title: "Attention Is All You Need",
                url: "https://arxiv.org/abs/1706.03762",
                type: "article",
                source: "arXiv"
              },
              {
                title: "Illustrated Transformer",
                url: "https://jalammar.github.io/illustrated-transformer/",
                type: "article",
                source: "Jay Alammar"
              }
            ],
            practice: [
              "Implement attention mechanism",
              "Build self-attention layer",
              "Create simple transformer block",
              "Apply to sequence-to-sequence task"
            ],
            timeEstimate: "8-9 hours"
          }
        ]
      },
      {
        week: 4,
        title: "Deep Learning Project",
        days: [
          {
            day: 47,
            title: "Month 4 Capstone - Deep Learning Project",
            description: "Build a complete deep learning application",
            topics: ["Model Architecture", "Training", "Deployment", "Documentation"],
            objectives: [
              "Design custom neural network architecture",
              "Train and optimize deep learning model",
              "Deploy model for inference",
              "Create comprehensive project documentation"
            ],
            resources: [
              {
                title: "PyTorch Project Template",
                url: "https://github.com/victoresque/pytorch-template",
                type: "tutorial",
                source: "GitHub"
              },
              {
                title: "Model Deployment Guide",
                url: "https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html",
                type: "tutorial",
                source: "PyTorch"
              }
            ],
            practice: [
              "Choose and implement project (image classification, NLP, etc.)",
              "Train deep learning model",
              "Optimize hyperparameters",
              "Deploy as web API",
              "Create project presentation"
            ],
            timeEstimate: "12-15 hours"
          }
        ]
      }
    ]
  },
  {
    month: 5,
    title: "Advanced AI & Specialization",
    weeks: [
      {
        week: 1,
        title: "Natural Language Processing",
        days: [
          {
            day: 48,
            title: "NLP Fundamentals & Word Embeddings",
            description: "Text processing and embeddings",
            topics: ["Tokenization", "Word2Vec", "GloVe", "FastText"],
            objectives: [
              "Master text preprocessing techniques",
              "Understand word embedding concepts",
              "Implement Word2Vec",
              "Use pre-trained embeddings"
            ],
            resources: [
              {
                title: "Word2Vec Tutorial",
                url: "https://www.youtube.com/watch?v=viZrOnJclY0",
                type: "video",
                source: "YouTube"
              },
              {
                title: "NLP Course - Hugging Face",
                url: "https://huggingface.co/learn/nlp-course",
                type: "course",
                source: "Hugging Face"
              }
            ],
            practice: [
              "Implement text preprocessing pipeline",
              "Train Word2Vec on corpus",
              "Visualize word embeddings",
              "Build document classifier"
            ],
            timeEstimate: "7-8 hours"
          },
          {
            day: 49,
            title: "Transformers & BERT",
            description: "Modern NLP with transformers",
            topics: ["Transformer Architecture", "BERT", "Fine-tuning", "Hugging Face"],
            objectives: [
              "Master Transformer architecture",
              "Use BERT for various NLP tasks",
              "Fine-tune pre-trained models",
              "Use Hugging Face library"
            ],
            resources: [
              {
                title: "BERT Explained",
                url: "https://jalammar.github.io/illustrated-bert/",
                type: "article",
                source: "Jay Alammar"
              },
              {
                title: "Hugging Face Transformers",
                url: "https://huggingface.co/docs/transformers/index",
                type: "documentation",
                source: "Hugging Face"
              }
            ],
            practice: [
              "Fine-tune BERT for classification",
              "Build question-answering system",
              "Implement sentiment analysis",
              "Create NLP pipeline"
            ],
            timeEstimate: "8-9 hours"
          }
        ]
      },
      {
        week: 2,
        title: "Computer Vision Advanced",
        days: [
          {
            day: 50,
            title: "Object Detection & Segmentation",
            description: "Advanced computer vision tasks",
            topics: ["YOLO", "R-CNN", "Semantic Segmentation", "Instance Segmentation"],
            objectives: [
              "Understand object detection architectures",
              "Implement YOLO and R-CNN variants",
              "Learn segmentation techniques",
              "Apply to real-world vision tasks"
            ],
            resources: [
              {
                title: "YOLO Object Detection",
                url: "https://www.youtube.com/watch?v=MPU2HistivI",
                type: "video",
                source: "YouTube"
              },
              {
                title: "Object Detection Guide",
                url: "https://pytorch.org/vision/stable/models.html",
                type: "documentation",
                source: "PyTorch"
              }
            ],
            practice: [
              "Implement object detection model",
              "Train on custom dataset",
              "Build image segmentation model",
              "Create real-time detection system"
            ],
            timeEstimate: "8-9 hours"
          },
          {
            day: 51,
            title: "Generative Models - GANs",
            description: "Generative Adversarial Networks",
            topics: ["GAN Architecture", "Training GANs", "DCGAN", "StyleGAN"],
            objectives: [
              "Understand GAN training dynamics",
              "Implement basic GAN",
              "Build DCGAN for images",
              "Explore advanced GAN variants"
            ],
            resources: [
              {
                title: "GAN Tutorial",
                url: "https://www.youtube.com/watch?v=8L11aMN5KY8",
                type: "video",
                source: "YouTube"
              },
              {
                title: "GAN Lab",
                url: "https://poloclub.github.io/ganlab/",
                type: "article",
                source: "Polo Club"
              }
            ],
            practice: [
              "Implement vanilla GAN",
              "Build DCGAN for image generation",
              "Train on custom dataset",
              "Explore latent space"
            ],
            timeEstimate: "8-9 hours"
          }
        ]
      },
      {
        week: 3,
        title: "Reinforcement Learning",
        days: [
          {
            day: 52,
            title: "RL Fundamentals",
            description: "Introduction to reinforcement learning",
            topics: ["MDP", "Q-Learning", "Policy Gradient", "Deep Q-Networks"],
            objectives: [
              "Understand RL problem formulation",
              "Implement Q-learning",
              "Learn policy gradient methods",
              "Build Deep Q-Network"
            ],
            resources: [
              {
                title: "Reinforcement Learning - David Silver",
                url: "https://www.youtube.com/watch?v=2pWv7GOvuf0",
                type: "video",
                source: "YouTube"
              },
              {
                title: "Spinning Up in Deep RL",
                url: "https://spinningup.openai.com/",
                type: "course",
                source: "OpenAI"
              }
            ],
            practice: [
              "Implement Q-learning for grid world",
              "Build DQN for Atari games",
              "Train agent in OpenAI Gym",
              "Visualize learning progress"
            ],
            timeEstimate: "8-9 hours"
          },
          {
            day: 53,
            title: "MLOps & Production AI",
            description: "Deploying AI systems in production",
            topics: ["MLOps", "Model Serving", "Monitoring", "CI/CD for ML"],
            objectives: [
              "Understand MLOps principles",
              "Deploy models to production",
              "Monitor model performance",
              "Implement ML pipelines"
            ],
            resources: [
              {
                title: "MLOps Guide",
                url: "https://madewithml.com/courses/mlops/",
                type: "course",
                source: "Made With ML"
              },
              {
                title: "ML Engineering Best Practices",
                url: "https://www.youtube.com/watch?v=pvaIi0l1GME",
                type: "video",
                source: "YouTube"
              }
            ],
            practice: [
              "Deploy model as REST API",
              "Set up model monitoring",
              "Create ML pipeline with MLflow",
              "Implement A/B testing"
            ],
            timeEstimate: "7-8 hours"
          }
        ]
      },
      {
        week: 4,
        title: "Final Project & Career Prep",
        days: [
          {
            day: 54,
            title: "Final Capstone Project - Day 1",
            description: "Start comprehensive AI project",
            topics: ["Project Planning", "Architecture Design", "Data Collection"],
            objectives: [
              "Design end-to-end AI system",
              "Plan project architecture",
              "Collect and prepare data",
              "Set up development environment"
            ],
            resources: [
              {
                title: "AI Project Ideas",
                url: "https://www.youtube.com/watch?v=tiwRp_FpGv0",
                type: "video",
                source: "YouTube"
              },
              {
                title: "Project Planning Guide",
                url: "https://towardsdatascience.com/how-to-plan-and-execute-machine-learning-projects-51dc5c9ffd3b",
                type: "article",
                source: "Towards Data Science"
              }
            ],
            practice: [
              "Choose ambitious AI project",
              "Design system architecture",
              "Set up Git repository",
              "Collect and explore data"
            ],
            timeEstimate: "8-10 hours"
          },
          {
            day: 55,
            title: "Final Capstone Project - Day 2",
            description: "Build and train models",
            topics: ["Model Development", "Training", "Optimization", "Evaluation"],
            objectives: [
              "Implement AI models",
              "Train and optimize",
              "Evaluate performance",
              "Iterate on improvements"
            ],
            resources: [
              {
                title: "Model Training Best Practices",
                url: "https://pytorch.org/tutorials/beginner/saving_loading_models.html",
                type: "tutorial",
                source: "PyTorch"
              }
            ],
            practice: [
              "Build model architecture",
              "Train models with hyperparameter tuning",
              "Evaluate on test set",
              "Optimize performance"
            ],
            timeEstimate: "10-12 hours"
          },
          {
            day: 56,
            title: "Final Capstone Project - Day 3",
            description: "Deploy and document project",
            topics: ["Deployment", "Documentation", "Portfolio", "Presentation"],
            objectives: [
              "Deploy AI system",
              "Create comprehensive documentation",
              "Prepare portfolio presentation",
              "Plan next steps in AI career"
            ],
            resources: [
              {
                title: "AI Portfolio Guide",
                url: "https://towardsdatascience.com/how-to-build-a-data-science-portfolio-5f566517c79c",
                type: "article",
                source: "Towards Data Science"
              },
              {
                title: "Model Deployment Tutorial",
                url: "https://www.youtube.com/watch?v=mrExsjcvF4o",
                type: "video",
                source: "YouTube"
              }
            ],
            practice: [
              "Deploy final project",
              "Write comprehensive README",
              "Create demo video",
              "Prepare portfolio presentation",
              "Celebrate completion of 5-month journey!"
            ],
            timeEstimate: "10-12 hours"
          }
        ]
      }
    ]
  }
];
