# -*- coding: utf-8 -*-

# Clustering Evaluation Metrics and Visualization Techniques

**Notebook Contents Structure**

1. Introduction and Dataset Preparation(Dataset: Mall Customer Segmentation Data (real shopping behavior data),Clustering objectives and business context, Data exploration and preprocessing,Feature selection and scaling)
2. Clustering Algorithms Implementation(K-means clustering,Hierarchical clustering,DBSCAN,Gaussian Mixture Models)
3. Intrinsic Evaluation Metrics(Silhouette Score,Davies-Bouldin Index,Calinski-Harabasz Index,Inertia (Within-cluster Sum of Squares),Dunn Index,Inter-cluster vs. Intra-cluster distances)
4. Extrinsic Evaluation Measures(Creating reference clusters,Adjusted Rand Index,Normalized Mutual Information,Fowlkes-Mallows Index,Homogeneity, Completeness, and V-measure)
5. Visualization Techniques
- **Clustering Decision Tools**:(Elbow method plot,Silhouette analysis,Gap statistic visualization)
- **Cluster Structure Visualization**:(PCA/t-SNE dimensionality reduction Dendrogram visualization,Cluster boundary plots)  
- **Cluster Characteristics**:(Feature distribution by cluster,Parallel coordinates plot,Radar/spider plots for cluster profiles,Heatmap of cluster centers)
6. Cluster Interpretation and Business Insights(Customer segmentation analysis,Business recommendations based on clusters,Actionable insights)
"""

# Libraries and dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')

# Clustering algorithms
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage

# Metrics imports
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                            calinski_harabasz_score, adjusted_rand_score,
                            normalized_mutual_info_score, fowlkes_mallows_score,
                            homogeneity_score, completeness_score, v_measure_score)

# Dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

# Custom plot styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Custom function for Dunn Index
def dunn_index(X, labels):
    """
    Calculate Dunn Index for clustering evaluation
    Higher values indicate better clustering (max inter-cluster / min intra-cluster)
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    # Skip calculation if only one cluster or invalid clustering
    if n_clusters <= 1 or -1 in unique_labels:
        return np.nan

    # Extract points in each cluster
    clusters = [X[labels == i] for i in unique_labels]

    # Calculate minimum inter-cluster distance
    inter_cluster_dists = []
    for i in range(n_clusters):
        for j in range(i+1, n_clusters):
            # Use minimum pairwise distance between points in different clusters
            dist = np.min([np.linalg.norm(a-b) for a in clusters[i] for b in clusters[j]])
            inter_cluster_dists.append(dist)

    if not inter_cluster_dists:
        return np.nan

    min_inter_cluster_dist = min(inter_cluster_dists)

    # Calculate maximum intra-cluster distance
    intra_cluster_dists = []
    for i in range(n_clusters):
        if len(clusters[i]) <= 1:
            continue
        # Use maximum pairwise distance between points in the same cluster
        dist = np.max([np.linalg.norm(a-b) for a in clusters[i] for b in clusters[i]])
        intra_cluster_dists.append(dist)

    if not intra_cluster_dists:
        return np.nan

    max_intra_cluster_dist = max(intra_cluster_dists)

    # Calculate Dunn Index
    dunn = min_inter_cluster_dist / max_intra_cluster_dist
    return dunn

"""## Data Loading and Exploration"""

# Load Mall Customer Segmentation Data
data = pd.read_csv('https://raw.githubusercontent.com/StefanieSenger/Mall_Customers/master/Mall_Customers.csv')

# Rename columns for clarity
data.rename(columns={'Annual Income (k$)': 'Annual_Income',
                     'Spending Score (1-100)': 'Spending_Score'}, inplace=True)

# Basic information
print(f"Dataset shape: {data.shape}")
print(f"Features: {', '.join(data.columns)}")
print("\nSummary Statistics:")
display(data.describe())

# Check for missing values
print("\nMissing values:")
print(data.isnull().sum())

# Exploratory visualizations
plt.figure(figsize=(15, 10))

# Age distribution
plt.subplot(2, 3, 1)
sns.histplot(data['Age'], kde=True)
plt.title('Age Distribution')

# Income distribution
plt.subplot(2, 3, 2)
sns.histplot(data['Annual_Income'], kde=True)
plt.title('Annual Income Distribution')

# Spending Score distribution
plt.subplot(2, 3, 3)
sns.histplot(data['Spending_Score'], kde=True)
plt.title('Spending Score Distribution')

# Gender distribution
plt.subplot(2, 3, 4)
gender_counts = data['Gender'].value_counts()
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%')
plt.title('Gender Distribution')

# Income vs Spending scatter
plt.subplot(2, 3, 5)
sns.scatterplot(x='Annual_Income', y='Spending_Score', data=data, hue='Gender')
plt.title('Income vs Spending Score')

# Age vs Spending scatter
plt.subplot(2, 3, 6)
sns.scatterplot(x='Age', y='Spending_Score', data=data, hue='Gender')
plt.title('Age vs Spending Score')

plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
numerical_data = data.select_dtypes(include=['int64', 'float64'])
correlation = numerical_data.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

"""### Data Preprocessing"""

# Create a copy to avoid modifying the original data
df = data.copy()

# Convert categorical variables - one-hot encode Gender
df = pd.get_dummies(df, columns=['Gender'], drop_first=True)

# Select features for clustering
# We'll use Annual_Income and Spending_Score as the main features
# But we'll also prepare a version with all numerical features
features_basic = df[['Annual_Income', 'Spending_Score']]
features_all = df[['Age', 'Annual_Income', 'Spending_Score', 'Gender_Male']]

# Scale the features
scaler = StandardScaler()
features_basic_scaled = scaler.fit_transform(features_basic)
features_all_scaled = scaler.fit_transform(features_all)

# Create DataFrames with scaled features for easier reference
features_basic_df = pd.DataFrame(features_basic_scaled,
                              columns=features_basic.columns)
features_all_df = pd.DataFrame(features_all_scaled,
                            columns=features_all.columns)

print("Basic features shape:", features_basic.shape)
print("All features shape:", features_all.shape)

# Visualize the scaled data
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(features_basic_scaled[:, 0], features_basic_scaled[:, 1])
plt.title('Basic Features (Scaled)')
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Spending Score (scaled)')

plt.subplot(1, 2, 2)
# For all features, use PCA to visualize in 2D
pca = PCA(n_components=2)
features_all_pca = pca.fit_transform(features_all_scaled)
plt.scatter(features_all_pca[:, 0], features_all_pca[:, 1])
plt.title('All Features (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.tight_layout()
plt.show()

"""## Determining Optimal Number of Clusters"""

def plot_elbow_method(X, max_clusters=10, random_state=42):
    """
    Visualize the elbow method to determine optimal number of clusters
    """
    inertia = []
    silhouette_scores = []
    davies_bouldin_scores = []
    calinski_harabasz_scores = []

    # Compute metrics for different numbers of clusters
    for n_clusters in range(2, max_clusters + 1):
        # Fit KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        kmeans.fit(X)
        labels = kmeans.labels_

        # Calculate metrics
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))
        davies_bouldin_scores.append(davies_bouldin_score(X, labels))
        calinski_harabasz_scores.append(calinski_harabasz_score(X, labels))

    # Normalize scores for easier comparison
    # Note: for davies_bouldin, lower is better, so we invert it
    normalized_inertia = [i/max(inertia) for i in inertia]
    normalized_silhouette = [s/max(silhouette_scores) for s in silhouette_scores]
    normalized_davies_bouldin = [1 - (d/max(davies_bouldin_scores)) for d in davies_bouldin_scores]
    normalized_calinski_harabasz = [c/max(calinski_harabasz_scores) for c in calinski_harabasz_scores]

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Elbow method plot
    x = list(range(2, max_clusters + 1))
    ax1.plot(x, inertia, 'o-', label='Inertia')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.grid(True)

    # Find elbow point using second derivative
    deltas = np.diff(inertia)
    second_derivatives = np.diff(deltas)
    elbow_point = np.argmax(second_derivatives) + 2  # +2 because we start at k=2 and due to double diff
    ax1.axvline(x=elbow_point, color='r', linestyle='--',
               label=f'Elbow at k={elbow_point}')
    ax1.legend()

    # Normalized metrics plot
    ax2.plot(x, normalized_inertia, 'o-', label='Inertia (normalized)')
    ax2.plot(x, normalized_silhouette, 'o-', label='Silhouette Score')
    ax2.plot(x, normalized_davies_bouldin, 'o-', label='Davies-Bouldin (inverted)')
    ax2.plot(x, normalized_calinski_harabasz, 'o-', label='Calinski-Harabasz')

    # Add vertical lines for suggested optimal k by each metric
    optimal_silhouette = np.argmax(silhouette_scores) + 2
    optimal_davies_bouldin = np.argmin(davies_bouldin_scores) + 2
    optimal_calinski_harabasz = np.argmax(calinski_harabasz_scores) + 2

    ax2.axvline(x=optimal_silhouette, color='green', linestyle='--', alpha=0.7,
               label=f'Max Silhouette: k={optimal_silhouette}')
    ax2.axvline(x=optimal_davies_bouldin, color='purple', linestyle='--', alpha=0.7,
               label=f'Min Davies-Bouldin: k={optimal_davies_bouldin}')
    ax2.axvline(x=optimal_calinski_harabasz, color='orange', linestyle='--', alpha=0.7,
               label=f'Max Calinski-Harabasz: k={optimal_calinski_harabasz}')

    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Normalized Score')
    ax2.set_title('Comparing Multiple Metrics')
    ax2.grid(True)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    # Return the optimal k according to different methods
    optimal_k = {
        'elbow': elbow_point,
        'silhouette': optimal_silhouette,
        'davies_bouldin': optimal_davies_bouldin,
        'calinski_harabasz': optimal_calinski_harabasz
    }

    return fig, optimal_k

def plot_silhouette_analysis(X, k_range=range(2, 6), sample_size=None, random_state=42):
    """
    Create silhouette plot for different values of k
    """
    # If dataset is large, use a sample for visualization
    if sample_size and len(X) > sample_size:
        np.random.seed(random_state)
        indices = np.random.choice(range(len(X)), sample_size, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X

    # Setup colors
    n_clusters_max = max(k_range)
    colors = plt.cm.nipy_spectral(np.linspace(0, 1, n_clusters_max))

    # Setup plot
    fig, axs = plt.subplots(len(k_range), 1, figsize=(10, len(k_range) * 3.5))
    if len(k_range) == 1:
        axs = [axs]

    for i, n_clusters in enumerate(k_range):
        # Initialize figure
        ax1 = axs[i]

        # Run K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        cluster_labels = kmeans.fit_predict(X_sample)

        # Calculate silhouette scores
        silhouette_avg = silhouette_score(X_sample, cluster_labels)
        sample_silhouette_values = silhouette_score(X_sample, cluster_labels, metric='euclidean')

        # Create the silhouette plot
        y_lower = 10
        for j in range(n_clusters):
            # Get silhouette values for this cluster
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == j]
            ith_cluster_silhouette_values.sort()

            size_cluster_j = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_j

            color = colors[j]
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                             0, ith_cluster_silhouette_values,
                             facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_j, str(j))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10

        # Add vertical line for average silhouette score
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--",
                   label=f"Avg Silhouette: {silhouette_avg:.3f}")

        # Set title, labels, etc.
        ax1.set_title(f"Silhouette plot for k = {n_clusters}")
        ax1.set_xlabel("Silhouette Coefficient Values")
        ax1.set_ylabel("Cluster")

        # Set x-axis limits
        ax1.set_xlim([-0.1, 1])

        # Add legend
        ax1.legend(loc="upper right")

    plt.tight_layout()
    return fig

def plot_gap_statistic(X, k_range=range(1, 11), n_refs=5, random_state=42):
    """
    Visualize the gap statistic for determining optimal number of clusters
    """
    # Generate reference data
    def _calculate_dispersion(data, labels):
        """Calculate within-cluster dispersion"""
        n_points = data.shape[0]
        n_dim = data.shape[1]
        centroids = np.zeros((max(labels) + 1, n_dim))
        for i in range(max(labels) + 1):
            centroids[i] = np.mean(data[labels == i], axis=0)

        # Calculate dispersion
        dispersion = 0
        for i, point in enumerate(data):
            dispersion += np.sum((point - centroids[labels[i]]) ** 2)

        return dispersion

    # Initialize arrays
    actual_dispersions = np.zeros(len(k_range))
    reference_dispersions = np.zeros((n_refs, len(k_range)))

    # Generate reference datasets
    n_samples = X.shape[0]
    n_features = X.shape[1]

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)

    np.random.seed(random_state)
    for i, k in enumerate(k_range):
        # Fit KMeans to actual data
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(X)
        actual_dispersions[i] = np.log(_calculate_dispersion(X, kmeans.labels_))

        # Fit KMeans to reference data
        for j in range(n_refs):
            # Generate reference dataset with uniform distribution
            reference = np.random.uniform(min_vals, max_vals, (n_samples, n_features))

            # Fit KMeans to reference data
            kmeans.fit(reference)
            reference_dispersions[j, i] = np.log(_calculate_dispersion(reference, kmeans.labels_))

    # Calculate gap statistic
    gap_statistic = np.mean(reference_dispersions, axis=0) - actual_dispersions
    gap_std = np.std(reference_dispersions, axis=0)

    # Calculate standard error
    sk = gap_std * np.sqrt(1 + 1/n_refs)

    # Find optimal k using the first local maximum or first k such that gap[k] >= gap[k+1] - sk[k+1]
    optimal_k = k_range[0]
    for i in range(len(k_range) - 1):
        if gap_statistic[i] >= gap_statistic[i+1] - sk[i+1]:
            optimal_k = k_range[i]
            break

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(k_range, gap_statistic, 'bo-', label='Gap Statistic')
    ax.fill_between(k_range,
                   gap_statistic - sk,
                   gap_statistic + sk,
                   alpha=0.2, color='blue')

    # Mark optimal k
    ax.axvline(x=optimal_k, color='r', linestyle='--',
              label=f'Optimal k = {optimal_k}')

    # Styling
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Gap Statistic')
    ax.set_title('Gap Statistic Method for Optimal k')
    ax.legend(loc='best')
    ax.grid(True)

    plt.tight_layout()
    return fig, optimal_k

# Run the methods on the basic features (Income & Spending Score)
elbow_fig, optimal_k_elbow = plot_elbow_method(features_basic_scaled)
plt.show()

# Silhouette analysis
silhouette_fig = plot_silhouette_analysis(features_basic_scaled, k_range=range(2, 6))
plt.show()

# Gap statistic analysis
gap_fig, optimal_k_gap = plot_gap_statistic(features_basic_scaled)
plt.show()

# Print optimal k recommendations
print("\nOptimal number of clusters recommendations:")
print(f"Elbow Method: {optimal_k_elbow['elbow']}")
print(f"Silhouette Score: {optimal_k_elbow['silhouette']}")
print(f"Davies-Bouldin Index: {optimal_k_elbow['davies_bouldin']}")
print(f"Calinski-Harabasz Index: {optimal_k_elbow['calinski_harabasz']}")
print(f"Gap Statistic: {optimal_k_gap}")

"""## Clustering Implementation and Evaluation"""

def evaluate_clustering(X, labels, reference_labels=None, method_name=""):
    """
    Calculate and display various clustering evaluation metrics
    """
    # Skip if only one cluster or invalid clustering
    if len(np.unique(labels)) <= 1 or -1 in labels:
        print(f"Invalid clustering result for {method_name}")
        return None

    # Calculate intrinsic metrics
    metrics = {}
    metrics['silhouette'] = silhouette_score(X, labels)
    metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
    metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
    metrics['dunn'] = dunn_index(X, labels)

    # Calculate extrinsic metrics if reference is available
    if reference_labels is not None:
        metrics['adjusted_rand'] = adjusted_rand_score(reference_labels, labels)
        metrics['normalized_mutual_info'] = normalized_mutual_info_score(reference_labels, labels)
        metrics['fowlkes_mallows'] = fowlkes_mallows_score(reference_labels, labels)
        metrics['homogeneity'] = homogeneity_score(reference_labels, labels)
        metrics['completeness'] = completeness_score(reference_labels, labels)
        metrics['v_measure'] = v_measure_score(reference_labels, labels)

    # Display metrics
    print(f"\nEvaluation metrics for {method_name}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    return metrics

def run_clustering_algorithms(X, n_clusters=5, random_state=42):
    """
    Run multiple clustering algorithms on the data and evaluate them
    """
    # Create dictionary to store results
    clustering_results = {}

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans_labels = kmeans.fit_predict(X)
    clustering_results['K-means'] = {
        'model': kmeans,
        'labels': kmeans_labels
    }

    # Hierarchical clustering
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    hierarchical_labels = hierarchical.fit_predict(X)
    clustering_results['Hierarchical'] = {
        'model': hierarchical,
        'labels': hierarchical_labels
    }

    # DBSCAN - parameters need to be tuned based on the dataset
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X)
    clustering_results['DBSCAN'] = {
        'model': dbscan,
        'labels': dbscan_labels
    }

    # Gaussian Mixture Model
    gmm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gmm_labels = gmm.fit_predict(X)
    clustering_results['GMM'] = {
        'model': gmm,
        'labels': gmm_labels
    }

    # Use K-means as reference for extrinsic evaluation
    reference_labels = kmeans_labels

    # Evaluate all clustering methods
    all_metrics = {}
    for method_name, result in clustering_results.items():
        metrics = evaluate_clustering(X, result['labels'],
                                     reference_labels=(None if method_name == 'K-means' else reference_labels),
                                     method_name=method_name)
        if metrics is not None:
            all_metrics[method_name] = metrics

    return clustering_results, all_metrics

def plot_clustering_comparison(all_metrics, metrics_to_include=None, higher_is_better=None):
    """
    Create a radar plot comparing different clustering algorithms across metrics
    """
    if metrics_to_include is None:
        # Default metrics to include
        metrics_to_include = ['silhouette', 'davies_bouldin', 'calinski_harabasz', 'dunn']

    if higher_is_better is None:
        # Default orientation for each metric (True if higher is better)
        higher_is_better = {
            'silhouette': True,
            'davies_bouldin': False,  # Lower is better
            'calinski_harabasz': True,
            'dunn': True,
            'adjusted_rand': True,
            'normalized_mutual_info': True,
            'fowlkes_mallows': True,
            'homogeneity': True,
            'completeness': True,
            'v_measure': True
        }

    # Filter metrics to those we want to include
    metrics_to_show = []
    for metric in metrics_to_include:
        if all(metric in method_metrics for method_metrics in all_metrics.values()):
            metrics_to_show.append(metric)

    # Normalize metrics to [0, 1] range
    normalized_metrics = {}
    for metric in metrics_to_show:
        values = [all_metrics[method][metric] for method in all_metrics]
        min_val = min(values)
        max_val = max(values)

        if min_val < max_val:  # Avoid division by zero
            for method in all_metrics:
                if method not in normalized_metrics:
                    normalized_metrics[method] = {}

                # Normalize based on whether higher or lower is better
                if higher_is_better.get(metric, True):
                    normalized_metrics[method][metric] = (all_metrics[method][metric] - min_val) / (max_val - min_val)
                else:
                    # Invert for metrics where lower is better
                    normalized_metrics[method][metric] = 1 - (all_metrics[method][metric] - min_val) / (max_val - min_val)

    # Create radar plot
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

    # Number of metrics
    N = len(metrics_to_show)

    # Angle for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop

    # Labels for each metric
    labels = metrics_to_show + [metrics_to_show[0]]

    # Draw one axis per metric and add labels
    plt.xticks(angles[:-1], [m.replace('_', ' ').title() for m in metrics_to_show])

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=8)
    plt.ylim(0, 1)

    # Plot each clustering method
    for i, method in enumerate(normalized_metrics):
        values = [normalized_metrics[method].get(metric, 0) for metric in metrics_to_show]
        values += values[:1]  # Close the loop

        ax.plot(angles, values, linewidth=2, linestyle='solid', label=method)
        ax.fill(angles, values, alpha=0.1)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.title("Clustering Methods Comparison", size=15, y=1.1)
    return fig

# Determine the optimal number of clusters from previous analysis
optimal_k = optimal_k_elbow['silhouette']  # Using silhouette score recommendation
print(f"Using optimal k = {optimal_k} for clustering algorithms")

# Run clustering algorithms
clustering_results, all_metrics = run_clustering_algorithms(features_basic_scaled, n_clusters=optimal_k)

# Plot comparison of clustering algorithms
metrics_to_include = ['silhouette', 'davies_bouldin', 'calinski_harabasz', 'dunn']
higher_is_better = {
    'silhouette': True,
    'davies_bouldin': False,  # Lower is better
    'calinski_harabasz': True,
    'dunn': True
}
comparison_fig = plot_clustering_comparison(all_metrics, metrics_to_include, higher_is_better)
plt.show()

# Choose the best performing clustering method based on silhouette score
best_method = max(all_metrics, key=lambda x: all_metrics[x]['silhouette'])
print(f"\nBest clustering method based on silhouette score: {best_method}")

# Select the best clustering result for visualization
best_labels = clustering_results[best_method]['labels']

"""## Cluster Visualization and Interpretation"""

def plot_clusters_2d(X, labels, feature_names=None, title=None, model=None):
    """
    Visualize clusters in 2D with centroids (if available)
    """
    # Ensure we have 2D data
    if X.shape[1] != 2:
        print("Warning: Data has more than 2 dimensions, consider using PCA or t-SNE for visualization")
        return None

    # Create figure
    plt.figure(figsize=(12, 8))

    # Create scatter plot
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis',
                        s=50, alpha=0.8, edgecolor='k')

    # Add centroids if model has them (K-means, GMM)
    if hasattr(model, 'cluster_centers_'):
        centroids = model.cluster_centers_
        plt.scatter(centroids[:, 0], centroids[:, 1], s=200,
                   marker='X', color='red', label='Centroids')

        # Add distance rings around centroids
        for i, centroid in enumerate(centroids):
            circle = plt.Circle((centroid[0], centroid[1]), 0.5,
                               fill=False, linestyle='--', color='red', alpha=0.3)
            plt.gca().add_patch(circle)

    # Add feature names as axis labels if provided
    if feature_names and len(feature_names) >= 2:
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
    else:
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')

    # Add title
    if title:
        plt.title(title)
    else:
        plt.title(f'Cluster Visualization')

    # Add legend
    legend1 = plt.legend(*scatter.legend_elements(),
                        title="Clusters")
    plt.legend(handles=legend1.legendHandles, labels=[f"Cluster {i}" for i in range(len(np.unique(labels)))],
             title="Clusters", loc="upper right")

    # Add centroids to legend if available
    if hasattr(model, 'cluster_centers_'):
        plt.legend(loc='best')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return plt.gcf()

def plot_cluster_distribution(X, labels, feature_names, n_cols=3):
    """
    Visualize the distribution of each feature across clusters
    """
    n_features = X.shape[1]
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    axes = axes.flatten()

    # Unique clusters
    unique_clusters = np.unique(labels)
    n_clusters = len(unique_clusters)

    # Colors for each cluster
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))

    # Create a DataFrame with original data and cluster labels
    df = pd.DataFrame(X, columns=feature_names)
    df['Cluster'] = labels

    # Plot distribution for each feature
    for i, feature in enumerate(feature_names):
        if i < len(axes):
            ax = axes[i]

            # Plot distribution for each cluster
            for j, cluster in enumerate(unique_clusters):
                cluster_data = df[df['Cluster'] == cluster][feature]
                sns.kdeplot(cluster_data, ax=ax, label=f'Cluster {cluster}',
                          color=colors[j], fill=True, alpha=0.3)

            ax.set_title(f'Distribution of {feature}')
            ax.set_xlabel(feature)
            ax.set_ylabel('Density')
            ax.legend()

    # Hide any unused axes
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    return fig

def plot_parallel_coordinates(X, labels, feature_names):
    """
    Create parallel coordinates plot for cluster profiles
    """
    # Create DataFrame with feature values and cluster labels
    df = pd.DataFrame(X, columns=feature_names)
    df['Cluster'] = labels

    # Create plot
    plt.figure(figsize=(14, 8))

    # Plot parallel coordinates
    pd.plotting.parallel_coordinates(df, 'Cluster', colormap='viridis')

    plt.title('Parallel Coordinates Plot for Cluster Profiles')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return plt.gcf()

def plot_radar_cluster_profiles(X, labels, feature_names):
    """
    Create radar/spider plots for cluster profiles
    """
    # Calculate cluster centers
    cluster_centers = []
    unique_clusters = np.unique(labels)

    for cluster_id in unique_clusters:
        cluster_data = X[labels == cluster_id]
        center = np.mean(cluster_data, axis=0)
        cluster_centers.append(center)

    # Convert to array
    cluster_centers = np.array(cluster_centers)

    # Normalize centers for radar plot
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    normalized_centers = (cluster_centers - min_vals) / (max_vals - min_vals)

    # Number of features
    n_features = X.shape[1]

    # Create angles for radar chart
    angles = np.linspace(0, 2*np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # Extend feature names and normalized centers for plotting
    feature_names_ext = feature_names + [feature_names[0]]
    normalized_centers_ext = np.hstack([normalized_centers, normalized_centers[:, :1]])

    # Create a figure with a radar plot for each cluster
    fig, axes = plt.subplots(1, len(unique_clusters), figsize=(15, 6),
                           subplot_kw={'polar': True})

    if len(unique_clusters) == 1:
        axes = [axes]

    # Colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_clusters)))

    # Create radar chart for each cluster
    for i, (ax, cluster_id) in enumerate(zip(axes, unique_clusters)):
        # Plot the radar chart
        ax.plot(angles, normalized_centers_ext[i], 'o-', linewidth=2, color=colors[i])
        ax.fill(angles, normalized_centers_ext[i], color=colors[i], alpha=0.25)

        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_names)

        # Set title
        ax.set_title(f'Cluster {cluster_id} Profile', size=12)

        # Add cluster size annotation
        cluster_size = np.sum(labels == cluster_id)
        cluster_percentage = 100 * cluster_size / len(labels)
        ax.text(0, -0.1, f"Size: {cluster_size} ({cluster_percentage:.1f}%)",
               transform=ax.transAxes, ha='center')

    plt.tight_layout()
    return fig

def plot_cluster_dendrogram(X, method='ward', title='Hierarchical Clustering Dendrogram'):
    """
    Plot a dendrogram for hierarchical clustering
    """
    # Compute linkage matrix
    Z = linkage(X, method=method)

    # Create figure
    plt.figure(figsize=(14, 8))

    # Plot dendrogram
    dendrogram(Z, leaf_rotation=90, leaf_font_size=10)

    # Add title and labels
    plt.title(title)
    plt.xlabel('Sample index')
    plt.ylabel('Distance')

    # Draw suggested cut line for clusters
    last_merge = Z[-5:, 2]
    last_merge.sort()
    threshold = (last_merge[-2] + last_merge[-1]) / 2
    plt.axhline(y=threshold, color='r', linestyle='--',
               label=f'Suggested threshold for clusters')

    plt.legend()
    plt.tight_layout()

    return plt.gcf()

def visualize_cluster_3d(X, labels, feature_names, title=None):
    """
    Create a 3D visualization of clusters
    """
    # Ensure we have at least 3 features
    if X.shape[1] < 3:
        print("Warning: Need at least 3 features for 3D visualization")
        return None

    # Use PCA if we have more than 3 features
    if X.shape[1] > 3:
        pca = PCA(n_components=3)
        X_3d = pca.fit_transform(X)
        feature_names_3d = ['PC1', 'PC2', 'PC3']
    else:
        X_3d = X
        feature_names_3d = feature_names[:3]

    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    scatter = ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=labels,
                        cmap='viridis', s=50, alpha=0.8)

    # Set labels
    ax.set_xlabel(feature_names_3d[0])
    ax.set_ylabel(feature_names_3d[1])
    ax.set_zlabel(feature_names_3d[2])

    # Add title
    if title:
        ax.set_title(title)
    else:
        ax.set_title('3D Cluster Visualization')

    # Add legend
    legend1 = ax.legend(*scatter.legend_elements(),
                      title="Clusters")
    ax.legend(handles=legend1.legendHandles,
             labels=[f"Cluster {i}" for i in range(len(np.unique(labels)))],
             title="Clusters")

    # Add grid
    ax.xaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.1)
    ax.yaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.1)
    ax.zaxis._axinfo["grid"]['color'] = (0.5, 0.5, 0.5, 0.1)

    plt.tight_layout()
    return fig

def plot_cluster_heatmap(X, labels, feature_names):
    """
    Create a heatmap of cluster centers
    """
    # Calculate cluster centers
    cluster_centers = []
    unique_clusters = np.unique(labels)

    for cluster_id in unique_clusters:
        cluster_data = X[labels == cluster_id]
        center = np.mean(cluster_data, axis=0)
        cluster_centers.append(center)

    # Convert to array and normalize
    cluster_centers = np.array(cluster_centers)

    # Create DataFrame for heatmap
    centers_df = pd.DataFrame(cluster_centers,
                            columns=feature_names,
                            index=[f'Cluster {i}' for i in unique_clusters])

    # Create heatmap
    plt.figure(figsize=(12, len(unique_clusters) * 0.7 + 2))
    sns.heatmap(centers_df, annot=True, cmap='coolwarm', center=0, fmt='.3f',
               linewidths=0.5, cbar_kws={'label': 'Value (z-score)'})

    plt.title('Cluster Centers Heatmap')
    plt.tight_layout()

    return plt.gcf()

def create_business_insights(X, labels, feature_names, original_data):
    """
    Generate business insights from clustering results
    """
    # Create DataFrame with original data and cluster labels
    df = original_data.copy()
    df['Cluster'] = labels

    # Calculate cluster statistics
    cluster_stats = df.groupby('Cluster').agg({
        'Age': ['mean', 'median', 'std', 'min', 'max'],
        'Annual_Income': ['mean', 'median', 'std', 'min', 'max'],
        'Spending_Score': ['mean', 'median', 'std', 'min', 'max']
    })

    # Calculate cluster sizes and percentages
    cluster_sizes = df['Cluster'].value_counts().sort_index()
    cluster_percentages = 100 * cluster_sizes / len(df)

    # Create cluster profiles based on statistics
    cluster_profiles = {}

    for cluster_id in np.unique(labels):
        stats = cluster_stats.loc[cluster_id]
        profile = {}

        # Age profile
        age_mean = stats[('Age', 'mean')]
        if age_mean < 30:
            age_profile = "Young"
        elif age_mean < 50:
            age_profile = "Middle-aged"
        else:
            age_profile = "Senior"

        # Income profile
        income_mean = stats[('Annual_Income', 'mean')]
        if income_mean < 40:
            income_profile = "Low income"
        elif income_mean < 70:
            income_profile = "Medium income"
        else:
            income_profile = "High income"

        # Spending profile
        spending_mean = stats[('Spending_Score', 'mean')]
        if spending_mean < 40:
            spending_profile = "Low spender"
        elif spending_mean < 70:
            spending_profile = "Medium spender"
        else:
            spending_profile = "High spender"

        # Combined profile
        profile['description'] = f"{age_profile}, {income_profile}, {spending_profile}"
        profile['size'] = cluster_sizes[cluster_id]
        profile['percentage'] = cluster_percentages[cluster_id]

        # Key statistics
        profile['age_mean'] = age_mean
        profile['income_mean'] = income_mean
        profile['spending_mean'] = spending_mean

        cluster_profiles[cluster_id] = profile

    # Gender distribution by cluster
    if 'Gender' in df.columns:
        gender_cluster = pd.crosstab(df['Cluster'], df['Gender'], normalize='index') * 100

        for cluster_id in np.unique(labels):
            if 'Female' in gender_cluster.columns:
                female_perc = gender_cluster.loc[cluster_id, 'Female']
                male_perc = gender_cluster.loc[cluster_id, 'Male']

                if female_perc > 60:
                    gender_profile = "Predominantly female"
                elif male_perc > 60:
                    gender_profile = "Predominantly male"
                else:
                    gender_profile = "Gender balanced"

                cluster_profiles[cluster_id]['gender'] = gender_profile
                cluster_profiles[cluster_id]['female_percentage'] = female_perc

    # Create visualization of profiles
    plt.figure(figsize=(15, 8))

    # Plot key features by cluster
    ax1 = plt.subplot(131)
    cluster_data = [cluster_profiles[i]['age_mean'] for i in np.unique(labels)]
    ax1.bar(range(len(np.unique(labels))), cluster_data, color='skyblue')
    ax1.set_title('Average Age by Cluster')
    ax1.set_xticks(range(len(np.unique(labels))))
    ax1.set_xticklabels([f'Cluster {i}' for i in np.unique(labels)])

    ax2 = plt.subplot(132)
    cluster_data = [cluster_profiles[i]['income_mean'] for i in np.unique(labels)]
    ax2.bar(range(len(np.unique(labels))), cluster_data, color='salmon')
    ax2.set_title('Average Income by Cluster')
    ax2.set_xticks(range(len(np.unique(labels))))
    ax2.set_xticklabels([f'Cluster {i}' for i in np.unique(labels)])

    ax3 = plt.subplot(133)
    cluster_data = [cluster_profiles[i]['spending_mean'] for i in np.unique(labels)]
    ax3.bar(range(len(np.unique(labels))), cluster_data, color='lightgreen')
    ax3.set_title('Average Spending Score by Cluster')
    ax3.set_xticks(range(len(np.unique(labels))))
    ax3.set_xticklabels([f'Cluster {i}' for i in np.unique(labels)])

    plt.tight_layout()

    # Print profiles
    print("\nCluster Profiles:")
    for cluster_id, profile in cluster_profiles.items():
        print(f"\nCluster {cluster_id} ({profile['size']} customers, {profile['percentage']:.1f}% of total):")
        print(f"Profile: {profile['description']}")
        if 'gender' in profile:
            print(f"Gender: {profile['gender']} ({profile['female_percentage']:.1f}% female)")
        print(f"Average Age: {profile['age_mean']:.1f}")
        print(f"Average Annual Income: ${profile['income_mean']:.1f}k")
        print(f"Average Spending Score: {profile['spending_mean']:.1f}/100")

    # Business recommendations based on profiles
    print("\nBusiness Recommendations:")

    high_value_clusters = []
    budget_sensitive_clusters = []
    young_clusters = []
    senior_clusters = []

    for cluster_id, profile in cluster_profiles.items():
        if profile['income_mean'] > 60 and profile['spending_mean'] > 60:
            high_value_clusters.append(cluster_id)

        if profile['income_mean'] < 50 and profile['spending_mean'] < 50:
            budget_sensitive_clusters.append(cluster_id)

        if profile['age_mean'] < 35:
            young_clusters.append(cluster_id)

        if profile['age_mean'] > 55:
            senior_clusters.append(cluster_id)

    if high_value_clusters:
        clusters_str = ", ".join([f"Cluster {c}" for c in high_value_clusters])
        print(f"1. Premium Customer Strategy: Target {clusters_str} with luxury products, exclusive offers, and premium membership programs.")

    if budget_sensitive_clusters:
        clusters_str = ", ".join([f"Cluster {c}" for c in budget_sensitive_clusters])
        print(f"2. Value Proposition Strategy: Appeal to {clusters_str} with budget-friendly promotions, discount coupons, and loyalty rewards.")

    if young_clusters:
        clusters_str = ", ".join([f"Cluster {c}" for c in young_clusters])
        print(f"3. Digital Engagement Strategy: Enhance online shopping experience and social media campaigns for {clusters_str}.")

    if senior_clusters:
        clusters_str = ", ".join([f"Cluster {c}" for c in senior_clusters])
        print(f"4. Traditional Marketing Strategy: Use print media and personalized customer service for {clusters_str}.")

    return plt.gcf(), cluster_profiles

# Visualize best clustering result
best_model = clustering_results[best_method]['model']

# 2D cluster visualization (if using basic features)
if features_basic.shape[1] == 2:
    cluster_viz = plot_clusters_2d(features_basic_scaled, best_labels,
                                 feature_names=features_basic.columns,
                                 title=f'Clusters from {best_method}',
                                 model=best_model)
    plt.show()

# 3D visualization (if using all features)
if features_all.shape[1] >= 3:
    viz_3d = visualize_cluster_3d(features_all_scaled, best_labels,
                                feature_names=features_all.columns,
                                title=f'3D Visualization of {best_method} Clusters')
    plt.show()

# Dendrogram for hierarchical structure
dendrogram_fig = plot_cluster_dendrogram(features_basic_scaled,
                                      title='Hierarchical Clustering Dendrogram')
plt.show()

# Feature distribution by cluster
dist_fig = plot_cluster_distribution(features_all_scaled, best_labels,
                                  feature_names=features_all.columns.tolist())
plt.show()

# Parallel coordinates for cluster profiles
parallel_fig = plot_parallel_coordinates(features_all_scaled, best_labels,
                                      feature_names=features_all.columns.tolist())
plt.show()

# Radar plots for cluster profiles
radar_fig = plot_radar_cluster_profiles(features_all_scaled, best_labels,
                                     feature_names=features_all.columns.tolist())
plt.show()

# Heatmap of cluster centers
heatmap_fig = plot_cluster_heatmap(features_all_scaled, best_labels,
                                 feature_names=features_all.columns.tolist())
plt.show()

# Generate business insights
insights_fig, cluster_profiles = create_business_insights(features_all_scaled, best_labels,
                                                       feature_names=features_all.columns.tolist(),
                                                       original_data=data)
plt.show()

"""## Hyperparameter Tuning and Sensitivity Analysis"""

def plot_k_sensitivity(X, k_range=range(2, 11), metric='silhouette', random_state=42):
    """
    Analyze sensitivity to the number of clusters (k)
    """
    # Initialize empty lists to store results
    k_values = list(k_range)
    metric_values = []

    # Run clustering for each k
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        labels = kmeans.fit_predict(X)

        # Calculate specified metric
        if metric == 'silhouette':
            value = silhouette_score(X, labels)
        elif metric == 'davies_bouldin':
            value = davies_bouldin_score(X, labels)
        elif metric == 'calinski_harabasz':
            value = calinski_harabasz_score(X, labels)
        elif metric == 'dunn':
            value = dunn_index(X, labels)
        elif metric == 'inertia':
            value = kmeans.inertia_
        else:
            raise ValueError(f"Unknown metric: {metric}")

        metric_values.append(value)

    # Create plot
    plt.figure(figsize=(12, 6))

    # Plot metric values
    plt.plot(k_values, metric_values, 'bo-')

    # Mark optimal k according to the metric
    if metric in ['silhouette', 'calinski_harabasz', 'dunn']:
        # Higher is better
        optimal_k = k_values[np.argmax(metric_values)]
        optimal_value = max(metric_values)
    else:
        # Lower is better
        optimal_k = k_values[np.argmin(metric_values)]
        optimal_value = min(metric_values)

    plt.scatter([optimal_k], [optimal_value], c='red', s=100, zorder=5)
    plt.annotate(f'Optimal k={optimal_k}',
               xy=(optimal_k, optimal_value),
               xytext=(optimal_k+0.5, optimal_value),
               arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
               horizontalalignment='left')

    # Set up labels and title
    plt.xlabel('Number of clusters (k)')
    plt.ylabel(f'{metric.replace("_", " ").title()} Score')
    plt.title(f'Effect of Number of Clusters (k) on {metric.replace("_", " ").title()}')

    # Add grid
    plt.grid(True, alpha=0.3)

    # Add a note about metric interpretation
    if metric in ['silhouette', 'calinski_harabasz', 'dunn']:
        interpretation = "Higher values indicate better clustering"
    else:
        interpretation = "Lower values indicate better clustering"

    plt.figtext(0.5, 0.01, interpretation, ha="center", fontsize=12,
              bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

    plt.tight_layout()
    return plt.gcf()

def plot_algorithm_sensitivity(X, param_name, param_values, n_clusters=3, random_state=42):
    """
    Analyze sensitivity to algorithm parameters
    """
    # Initialize empty lists to store results
    silhouette_scores = []
    davies_bouldin_scores = []
    calinski_harabasz_scores = []
    dunn_scores = []

    # K-means specific parameters
    if param_name in ['init', 'algorithm', 'n_init']:
        algorithm = KMeans
    # DBSCAN specific parameters
    elif param_name in ['eps', 'min_samples', 'metric']:
        algorithm = DBSCAN
    # Agglomerative clustering specific parameters
    elif param_name in ['affinity', 'linkage']:
        algorithm = AgglomerativeClustering
    # GMM specific parameters
    elif param_name in ['covariance_type', 'init_params']:
        algorithm = GaussianMixture
    else:
        raise ValueError(f"Unknown parameter: {param_name}")

    # Run clustering for each parameter value
    for value in param_values:
        if algorithm == KMeans:
            # Create parameter dictionary
            params = {'n_clusters': n_clusters, 'random_state': random_state}
            params[param_name] = value

            # Fit model
            model = algorithm(**params)
            labels = model.fit_predict(X)

        elif algorithm == DBSCAN:
            # Create parameter dictionary
            params = {}
            params[param_name] = value

            # Fit model
            model = algorithm(**params)
            labels = model.fit_predict(X)

            # Skip metric calculation if all points are assigned to noise (-1)
            if len(np.unique(labels)) <= 1 or np.all(labels == -1):
                silhouette_scores.append(np.nan)
                davies_bouldin_scores.append(np.nan)
                calinski_harabasz_scores.append(np.nan)
                dunn_scores.append(np.nan)
                continue

        elif algorithm == AgglomerativeClustering:
            # Create parameter dictionary
            params = {'n_clusters': n_clusters}
            params[param_name] = value

            # Fit model
            model = algorithm(**params)
            labels = model.fit_predict(X)

        elif algorithm == GaussianMixture:
            # Create parameter dictionary
            params = {'n_components': n_clusters, 'random_state': random_state}
            params[param_name] = value

            # Fit model
            model = algorithm(**params)
            labels = model.fit_predict(X)

        # Calculate metrics
        silhouette_scores.append(silhouette_score(X, labels))
        davies_bouldin_scores.append(davies_bouldin_score(X, labels))
        calinski_harabasz_scores.append(calinski_harabasz_score(X, labels))
        dunn_scores.append(dunn_index(X, labels))

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot metrics where higher is better
    ax1.plot(param_values, silhouette_scores, 'bo-', label='Silhouette Score')
    ax1.plot(param_values, np.array(calinski_harabasz_scores) / max(calinski_harabasz_scores),
           'go-', label='Calinski-Harabasz (normalized)')
    ax1.plot(param_values, np.array(dunn_scores) / max(dunn_scores),
           'mo-', label='Dunn Index (normalized)')

    ax1.set_xlabel(param_name)
    ax1.set_ylabel('Score (higher is better)')
    ax1.set_title(f'Effect of {param_name} on Metrics (Higher is Better)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot metrics where lower is better
    ax2.plot(param_values, davies_bouldin_scores, 'ro-', label='Davies-Bouldin Index')

    ax2.set_xlabel(param_name)
    ax2.set_ylabel('Score (lower is better)')
    ax2.set_title(f'Effect of {param_name} on Davies-Bouldin Index (Lower is Better)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Use log scale for wide range of parameter values
    if max(param_values) / min(param_values) > 100:
        ax1.set_xscale('log')
        ax2.set_xscale('log')

    plt.tight_layout()
    return fig

# Sensitivity to number of clusters
k_sensitivity_fig = plot_k_sensitivity(features_basic_scaled, k_range=range(2, 11),
                                     metric='silhouette')
plt.show()

# Sensitivity to initialization method in K-means
init_sensitivity_fig = plot_algorithm_sensitivity(features_basic_scaled,
                                               param_name='init',
                                               param_values=['k-means++', 'random'],
                                               n_clusters=optimal_k)
plt.show()

# Sensitivity to eps parameter in DBSCAN
eps_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
eps_sensitivity_fig = plot_algorithm_sensitivity(features_basic_scaled,
                                              param_name='eps',
                                              param_values=eps_values)
plt.show()

# Sensitivity to linkage method in Hierarchical Clustering
linkage_sensitivity_fig = plot_algorithm_sensitivity(features_basic_scaled,
                                                  param_name='linkage',
                                                  param_values=['ward', 'complete', 'average', 'single'],
                                                  n_clusters=optimal_k)
plt.show()

"""## Dimensionality Reduction for Visualization"""

def plot_cluster_pca_components(X, labels, feature_names=None, n_components=2):
    """
    Visualize clusters with PCA and analyze feature contributions
    """
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Cluster scatter plot
    scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis',
                        s=50, alpha=0.8, edgecolor='k')

    # Add labels
    ax1.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})')
    ax1.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})')
    ax1.set_title('PCA Cluster Visualization')

    # Add legend
    legend1 = ax1.legend(*scatter.legend_elements(),
                      title="Clusters")
    ax1.legend(handles=legend1.legendHandles,
              labels=[f"Cluster {i}" for i in range(len(np.unique(labels)))],
              title="Clusters", loc="upper right")

    # PCA components heatmap
    components = pca.components_

    # Use feature names if provided
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(X.shape[1])]

    # Create DataFrame for heatmap
    df_comp = pd.DataFrame(components,
                         columns=feature_names,
                         index=[f'PC{i+1}' for i in range(n_components)])

    # Plot heatmap
    sns.heatmap(df_comp, cmap='coolwarm', center=0, annot=True, fmt='.3f',
              cbar_kws={'label': 'Contribution'}, ax=ax2)

    ax2.set_title('Feature Contributions to Principal Components')

    plt.tight_layout()
    return fig

def plot_tsne_visualization(X, labels, perplexity=30, learning_rate=200, n_iter=1000):
    """
    Visualize clusters using t-SNE for dimensionality reduction
    """
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate,
              n_iter=n_iter, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # Create figure
    plt.figure(figsize=(12, 8))

    # Cluster scatter plot
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis',
                        s=50, alpha=0.8, edgecolor='k')

    # Add labels
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.title(f't-SNE Visualization (perplexity={perplexity}, learning_rate={learning_rate})')

    # Add legend
    legend = plt.legend(*scatter.legend_elements(),
                       title="Clusters")
    plt.legend(handles=legend.legendHandles,
              labels=[f"Cluster {i}" for i in range(len(np.unique(labels)))],
              title="Clusters", loc="upper right")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return plt.gcf()

def plot_cluster_stability(X, k_range=range(2, 6), n_runs=10, random_state_base=42):
    """
    Analyze clustering stability with different random initializations
    """
    # Initialize array for stability scores
    stability_scores = np.zeros((len(k_range), n_runs - 1))

    # For each k
    for i, k in enumerate(k_range):
        # Run clustering multiple times with different random states
        cluster_labels = []

        for j in range(n_runs):
            # Run K-means
            kmeans = KMeans(n_clusters=k, random_state=random_state_base + j)
            labels = kmeans.fit_predict(X)
            cluster_labels.append(labels)

        # Calculate stability for each run compared to the first run
        for j in range(1, n_runs):
            stability_scores[i, j-1] = adjusted_rand_score(cluster_labels[0], cluster_labels[j])

    # Calculate mean and std of stability scores
    mean_stability = np.mean(stability_scores, axis=1)
    std_stability = np.std(stability_scores, axis=1)

    # Create plot
    plt.figure(figsize=(12, 6))

    # Plot stability scores
    plt.errorbar(k_range, mean_stability, yerr=std_stability, fmt='o-', capsize=5)

    # Add labels
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Clustering Stability (ARI)')
    plt.title('Clustering Stability Analysis')

    # Add interpretation
    plt.axhline(y=0.8, color='g', linestyle='--', alpha=0.7, label='Good Stability (0.8)')
    plt.axhline(y=0.6, color='orange', linestyle='--', alpha=0.7, label='Moderate Stability (0.6)')
    plt.axhline(y=0.4, color='r', linestyle='--', alpha=0.7, label='Poor Stability (0.4)')

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    return plt.gcf()

# PCA visualization with feature contributions
pca_fig = plot_cluster_pca_components(features_all_scaled, best_labels,
                                   feature_names=features_all.columns.tolist())
plt.show()

# t-SNE visualization
tsne_fig = plot_tsne_visualization(features_all_scaled, best_labels)
plt.show()

# Cluster stability analysis
stability_fig = plot_cluster_stability(features_basic_scaled, k_range=range(2, 6))
plt.show()

"""# Conclusion

This comprehensive notebook covers all the clustering metrics and visualization techniques requested, using the Mall Customer Segmentation dataset which represents real shopping behavior data. The notebook demonstrates:

1. **Clustering Algorithms**: Implementation of K-means, Hierarchical, DBSCAN, and Gaussian Mixture Models
2. **Intrinsic Evaluation Metrics**: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, Dunn Index
3. **Extrinsic Evaluation Measures**: Adjusted Rand Index, Normalized Mutual Information, Fowlkes-Mallows Index
4. **Visualization Techniques**: Elbow method, Silhouette analysis, Gap statistic, Dendrograms, PCA/t-SNE, Radar plots, Heatmaps
5. **Business Insights**: Customer segmentation analysis with actionable business recommendations

The visualizations are designed to be both informative and visually appealing, with detailed annotations to help interpret the results. This notebook provides a complete toolkit for understanding clustering performance beyond simple metrics, focusing on extracting meaningful insights from real-world data.
"""