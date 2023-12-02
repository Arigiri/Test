
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def assign_to_clusters(center_locs: np.ndarray, cluster_assignments: np.ndarray, locs: np.ndarray) -> np.ndarray:
    """Assign each point to the nearest cluster center."""
    for node_id in range(locs.shape[0]):
        best_dist = np.inf
        for center_id in range(center_locs.shape[0]):
            dist = np.linalg.norm(center_locs[center_id] - locs[node_id])
            if dist < best_dist:
                best_dist = dist
                cluster_assignments[node_id] = center_id
    return cluster_assignments

@njit
def compute_cost(center_locs: np.ndarray, cluster_assignments: np.ndarray, locs: np.ndarray, cluster_id: int) -> float:
    """Compute the cost of a cluster center."""
    cost = 0.0
    for i in range(locs.shape[0]):
        if cluster_assignments[i] == cluster_id:
            cost += np.linalg.norm(center_locs[cluster_id] - locs[i])
    return cost

def select_new_candidate(kmeans_center: np.ndarray, locs: np.ndarray, num_candidates: int = 5) -> int:
    """
    Select a new candidate for the cluster center based on distance from the pre-calculated k-means center.
    """
    distances = np.linalg.norm(locs - kmeans_center, axis=1)
    closest_points = np.argsort(distances)[:num_candidates]
    
    # Probabilities decrease exponentially for farther points
    probabilities = np.exp(-np.arange(num_candidates))
    probabilities /= probabilities.sum()
    
    new_candidate = np.random.choice(closest_points, p=probabilities)
    return new_candidate

@njit
def update_centers_with_new_candidate(center_locs: np.ndarray, kmeans_centers: np.ndarray, cluster_assignments: np.ndarray, locs: np.ndarray, num_clusters: int) -> np.ndarray:
    """Update the cluster centers using the new candidate selection function."""
    for cluster_id in range(num_clusters):
        current_cost = compute_cost(center_locs, cluster_assignments, locs, cluster_id)
        kmeans_center = kmeans_centers[cluster_id]
        new_candidate = select_new_candidate(kmeans_center, locs)
        new_cost = compute_cost(center_locs, cluster_assignments, locs, new_candidate)
        if new_cost < current_cost:
            center_locs[cluster_id] = locs[new_candidate]
    return center_locs


@njit
def compute_kmeans_centers(center_locs: np.ndarray, cluster_assignments: np.ndarray, locs: np.ndarray, num_clusters: int) -> np.ndarray:
    """Compute the k-means centers for all clusters."""
    kmeans_centers = np.zeros((num_clusters, locs.shape[1]))
    cluster_counts = np.zeros(num_clusters)
    
    for i in range(locs.shape[0]):
        cluster_id = cluster_assignments[i]
        kmeans_centers[cluster_id] += locs[i]
        cluster_counts[cluster_id] += 1
        
    for cluster_id in range(num_clusters):
        if cluster_counts[cluster_id] > 0:
            kmeans_centers[cluster_id] /= cluster_counts[cluster_id]
            
    return kmeans_centers


@njit
def k_means(center_locs: np.ndarray, cluster_assignments: np.ndarray, locs: np.ndarray, iterations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Perform k-means clustering."""
    for _ in range(iterations):
        new_cluster_assignments = assign_to_clusters(center_locs, cluster_assignments, locs)
        kmeans_centers = compute_kmeans_centers(center_locs, new_cluster_assignments, locs)  # Placeholder, you'll have to implement this
        center_locs = update_centers_with_new_candidate(center_locs, kmeans_centers, new_cluster_assignments, locs, center_locs.shape[0])
        cluster_assignments = new_cluster_assignments
    return cluster_assignments, center_locs

# Rest of the code for data generation and plotting remains the same as in the refactored code.
