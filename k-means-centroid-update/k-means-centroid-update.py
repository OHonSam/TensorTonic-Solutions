import numpy as np

def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    points = np.array(points)
    assignments = np.array(assignments)
    centroids = []
    
    for cluster_idx in range(k):
        mask = (assignments == cluster_idx)
        total_assigned = np.sum(mask)
        points_sum = np.sum(points[mask], axis=0)
        
        if total_assigned != 0:
            centroid = points_sum / total_assigned
        else:
            centroid = np.zeros(points.shape[1])

        centroids.append(centroid.tolist())

    return centroids
    
            