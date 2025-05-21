# Visual Intelligence beyond Accuracy for AI
## Evaluation, Comparison & Optimisation of ML/DL Models
This repository explores how to visualise and evaluate AI model performance effectively, compare models through meaningful plots, choose the right metrics, and optimise parameters using modern tools. Thus, empowering you to make smarter, data-driven decisions in ML/DL projects.
Each section corresponds to topics covered in the seminar, with code examples designed to be both educational and adaptable to your own AI projects.

### Objectives:
  * Recap the foundamentals of ML and DL methods
  * Explore key evaluation metrics for classification and regression.
  * Compare multiple models using interactive visualisations (radar plots, ROC curves, etc.).
  * Understand and interpret model performance through confusion matrices, learning curves, and error distributions.
  * Use tools like Scikit-learn, Optuna, MLflow, TensorBoard, and Weights & Biases to tune parameters and track experiments visually.

### Content
* Visualization of classification metrics (Confusion matrices, ROC/PR curves)
* Regression error analysis through visual techniques
* Model comparison using parallel coordinates and radar plots
* Learning curve interpretation for detecting overfitting
* Hyperparameter optimization visualization
* Integration with tracking tools (MLflow, Weights & Biases)
* Best practices for visualization in academic reporting

### Target Audience
Data scientists, ML engineers, researchers, and practitioners who want to move beyond simple accuracy metrics to gain deeper insights into model performance and behavior through effective visualization techniques.

### Requirements
The notebook uses common Python libraries including matplotlib, seaborn, plotly, scikit-learn, and optional integrations with MLflow and Weights & Biases. Setup instructions are provided in the first code cell.

#### Libraries to Install
You can install all the required libraries using pip:
'''
bashpip install numpy pandas matplotlib seaborn scikit-learn tensorflow ipython
'''
If you're using a Jupyter notebook, you might already have IPython installed. For GPU support with TensorFlow, you might need to install a specific version:
'''
bashpip install tensorflow-gpu  # If you have a compatible NVIDIA GPU
'''
Note that this code seems to include components for different deep learning projects (computer vision with CIFAR-10, NLP with IMDB, image generation with Fashion-MNIST). You may not need all these imports for a single project.
