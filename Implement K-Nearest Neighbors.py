import numpy as np

def k_nearest_neighbors(points, query_point, k):
    # Step 1: compute (distance, point)
    dist_points = []

    xq, yq = query_point

    for x, y in points:
        dist = np.sqrt((x - xq)**2 + (y - yq)**2)
        dist_points.append((dist, (x, y)))

    # Step 2: sort by distance (stable sort keeps input order for ties)
    dist_points.sort(key=lambda x: x[0])

    # Step 3: take k nearest and return only points
    return [point for _, point in dist_points[:k]]