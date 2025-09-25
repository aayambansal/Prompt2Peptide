# --- embeddings.py ---
import torch
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
import esm

def get_esm_embeddings(sequences, model_name="esm2_t6_8M_UR50D", normalize=True):
    """Extract ESM-2 embeddings for sequences with optional L2 normalization"""
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    
    embeddings = []
    
    with torch.no_grad():
        for seq in sequences:
            # Prepare data
            data = [("seq", seq)]
            _, _, tokens = batch_converter(data)
            
            if torch.cuda.is_available():
                model = model.cuda()
                tokens = tokens.cuda()
            
            # Get embeddings (use last hidden state)
            results = model(tokens, repr_layers=[model.num_layers], return_contacts=False)
            token_embeddings = results["representations"][model.num_layers]
            
            # Average pool over sequence length (excluding CLS and SEP tokens)
            sequence_embedding = token_embeddings[0, 1:-1, :].mean(dim=0)
            
            # L2 normalization if requested
            if normalize:
                sequence_embedding = torch.nn.functional.normalize(sequence_embedding, p=2, dim=0)
            
            embeddings.append(sequence_embedding.cpu().numpy())
    
    return np.array(embeddings)

def compute_centroid_distances(embeddings, labels):
    """Compute distances between centroids of different prompt clusters"""
    unique_labels = np.unique(labels)
    centroids = {}
    
    for label in unique_labels:
        mask = labels == label
        centroids[label] = np.mean(embeddings[mask], axis=0)
    
    # Compute pairwise distances between centroids
    centroid_distances = {}
    for i, label1 in enumerate(unique_labels):
        for j, label2 in enumerate(unique_labels):
            if i < j:  # Avoid duplicates
                dist = np.linalg.norm(centroids[label1] - centroids[label2])
                centroid_distances[f"{label1}_vs_{label2}"] = dist
    
    return centroids, centroid_distances

def create_embedding_plot(sequences_dict, title="ESM-2 Embedding Space", method="umap"):
    """Create UMAP/t-SNE plot of ESM embeddings colored by prompt type"""
    
    # Combine all sequences with labels
    all_sequences = []
    all_labels = []
    all_prompt_types = []
    
    for prompt_type, sequences in sequences_dict.items():
        all_sequences.extend(sequences)
        all_labels.extend([prompt_type] * len(sequences))
        all_prompt_types.extend([prompt_type] * len(sequences))
    
    print(f"Computing ESM-2 embeddings for {len(all_sequences)} sequences...")
    
    # Get embeddings
    embeddings = get_esm_embeddings(all_sequences)
    
    # Dimensionality reduction
    if method.lower() == "umap":
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embedding_2d = reducer.fit_transform(embeddings)
    else:  # t-SNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)//4))
        embedding_2d = reducer.fit_transform(embeddings)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    for i, prompt_type in enumerate(np.unique(all_labels)):
        mask = np.array(all_labels) == prompt_type
        plt.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1], 
                   c=colors[i % len(colors)], label=prompt_type, alpha=0.7, s=50)
    
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Compute and display centroid distances
    centroids, centroid_distances = compute_centroid_distances(embedding_2d, np.array(all_labels))
    
    # Add centroid distance info to plot
    distance_text = "Centroid Distances:\n"
    for pair, dist in centroid_distances.items():
        distance_text += f"{pair}: {dist:.2f}\n"
    
    plt.text(0.02, 0.98, distance_text, transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=8, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return plt.gcf(), embedding_2d, centroids, centroid_distances

def embedding_analysis_summary(sequences_dict):
    """Comprehensive embedding analysis with silhouette scores"""
    print("🔍 EMBEDDING SPACE ANALYSIS")
    print("="*50)
    
    # Get embeddings for all sequences
    all_sequences = []
    all_labels = []
    
    for prompt_type, sequences in sequences_dict.items():
        all_sequences.extend(sequences)
        all_labels.extend([prompt_type] * len(sequences))
    
    print(f"Computing ESM-2 embeddings for {len(all_sequences)} sequences...")
    embeddings = get_esm_embeddings(all_sequences)
    
    # Compute silhouette score
    if len(np.unique(all_labels)) > 1:
        silhouette_avg = silhouette_score(embeddings, all_labels)
        print(f"\n📊 SILHOUETTE SCORE: {silhouette_avg:.3f}")
        
        # Per-cluster silhouette scores
        from sklearn.metrics import silhouette_samples
        silhouette_samples_scores = silhouette_samples(embeddings, all_labels)
        
        cluster_silhouettes = {}
        for i, label in enumerate(np.unique(all_labels)):
            mask = np.array(all_labels) == label
            cluster_silhouettes[label] = np.mean(silhouette_samples_scores[mask])
            print(f"  {label}: {cluster_silhouettes[label]:.3f}")
    else:
        silhouette_avg = np.nan
        cluster_silhouettes = {}
        print(f"\n📊 SILHOUETTE SCORE: N/A (only one cluster)")
    
    # Compute centroid distances
    centroids, centroid_distances = compute_centroid_distances(embeddings, np.array(all_labels))
    
    print(f"\n📊 CENTROID DISTANCES:")
    for pair, dist in centroid_distances.items():
        print(f"  {pair}: {dist:.3f}")
    
    # Compute within-cluster and between-cluster distances
    unique_labels = np.unique(all_labels)
    within_cluster_distances = []
    between_cluster_distances = []
    
    for label in unique_labels:
        mask = np.array(all_labels) == label
        cluster_embeddings = embeddings[mask]
        
        if len(cluster_embeddings) > 1:
            # Within-cluster distances
            within_distances = pdist(cluster_embeddings)
            within_cluster_distances.extend(within_distances)
        
        # Between-cluster distances
        for other_label in unique_labels:
            if other_label != label:
                other_mask = np.array(all_labels) == other_label
                other_embeddings = embeddings[other_mask]
                
                for emb1 in cluster_embeddings:
                    for emb2 in other_embeddings:
                        between_cluster_distances.append(np.linalg.norm(emb1 - emb2))
    
    avg_within = np.mean(within_cluster_distances) if within_cluster_distances else 0
    avg_between = np.mean(between_cluster_distances) if between_cluster_distances else 0
    
    print(f"\n📈 CLUSTER SEPARATION:")
    print(f"  Average within-cluster distance: {avg_within:.3f}")
    print(f"  Average between-cluster distance: {avg_between:.3f}")
    print(f"  Separation ratio: {avg_between/avg_within:.3f}" if avg_within > 0 else "  Separation ratio: N/A")
    
    return {
        'embeddings': embeddings,
        'centroids': centroids,
        'centroid_distances': centroid_distances,
        'within_cluster_avg': avg_within,
        'between_cluster_avg': avg_between,
        'separation_ratio': avg_between/avg_within if avg_within > 0 else np.nan,
        'silhouette_score': silhouette_avg,
        'cluster_silhouettes': cluster_silhouettes
    }

def create_comprehensive_embedding_plot(embeddings, labels, save_path='comprehensive_embedding_analysis.png'):
    """Create comprehensive embedding visualization with UMAP, t-SNE, and PCA"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comprehensive Embedding Analysis: UMAP, t-SNE, and PCA', 
                 fontsize=16, fontweight='bold')
    
    # Get unique labels and colors
    unique_labels = np.unique(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    label_colors = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    # 1. UMAP
    ax = axes[0, 0]
    try:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embedding_2d = reducer.fit_transform(embeddings)
        
        for label in unique_labels:
            mask = labels == label
            ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1], 
                      c=[label_colors[label]], label=label, alpha=0.7, s=50)
        
        ax.set_title('UMAP Projection')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    except Exception as e:
        ax.text(0.5, 0.5, f'UMAP failed: {str(e)}', ha='center', va='center', transform=ax.transAxes)
    
    # 2. t-SNE
    ax = axes[0, 1]
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)//4))
        embedding_tsne = tsne.fit_transform(embeddings)
        
        for label in unique_labels:
            mask = labels == label
            ax.scatter(embedding_tsne[mask, 0], embedding_tsne[mask, 1], 
                      c=[label_colors[label]], label=label, alpha=0.7, s=50)
        
        ax.set_title('t-SNE Projection')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    except Exception as e:
        ax.text(0.5, 0.5, f't-SNE failed: {str(e)}', ha='center', va='center', transform=ax.transAxes)
    
    # 3. PCA
    ax = axes[1, 0]
    try:
        pca = PCA(n_components=2, random_state=42)
        embedding_pca = pca.fit_transform(embeddings)
        
        for label in unique_labels:
            mask = labels == label
            ax.scatter(embedding_pca[mask, 0], embedding_pca[mask, 1], 
                      c=[label_colors[label]], label=label, alpha=0.7, s=50)
        
        ax.set_title(f'PCA Projection (Explained Variance: {pca.explained_variance_ratio_.sum():.3f})')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    except Exception as e:
        ax.text(0.5, 0.5, f'PCA failed: {str(e)}', ha='center', va='center', transform=ax.transAxes)
    
    # 4. Silhouette Analysis
    ax = axes[1, 1]
    try:
        from sklearn.metrics import silhouette_samples
        silhouette_samples_scores = silhouette_samples(embeddings, labels)
        
        y_lower = 10
        for i, label in enumerate(unique_labels):
            mask = labels == label
            cluster_silhouette_values = silhouette_samples_scores[mask]
            cluster_silhouette_values.sort()
            
            size_cluster = len(cluster_silhouette_values)
            y_upper = y_lower + size_cluster
            
            color = label_colors[label]
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_values,
                           facecolor=color, edgecolor=color, alpha=0.7)
            
            ax.text(-0.05, y_lower + 0.5 * size_cluster, str(label))
            y_lower = y_upper + 10
        
        ax.set_xlabel('Silhouette Coefficient')
        ax.set_ylabel('Cluster')
        ax.set_title('Silhouette Analysis')
        ax.axvline(x=0, color="black", linestyle="--")
        ax.grid(True, alpha=0.3)
    except Exception as e:
        ax.text(0.5, 0.5, f'Silhouette failed: {str(e)}', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig
