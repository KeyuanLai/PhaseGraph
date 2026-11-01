import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans

def cluster_matrix(frame, groups, dimensions=2, method="ward", seed=42):
    if list(frame.index.astype(str)) != list(frame.columns.astype(str)):
        raise ValueError("Matrix row and column labels must match in order")
    d = frame.to_numpy(float); n = len(d)
    if n < 2 or d.shape != (n,n) or not np.all(np.isfinite(d)) or np.any(d < -1e-12) or not np.allclose(d,d.T,atol=1e-10) or not np.allclose(np.diag(d),0,atol=1e-10):
        raise ValueError("Expected a finite nonnegative symmetric matrix with zero diagonal")
    if not 2 <= groups <= n or dimensions < 1 or method not in {"ward", "kmeans"}:
        raise ValueError("Invalid group count, dimension or clustering method")
    j = np.eye(n)-np.ones((n,n))/n
    values, vectors = np.linalg.eigh(-.5*j@(d*d)@j); order = np.argsort(values)[::-1]
    values = values[order]; vectors = vectors[:,order]; keep = values > max(1e-12,np.max(np.abs(values))*1e-12)
    if not np.any(keep):
        raise ValueError("No positive PCoA dimensions; the supplied dissimilarities do not separate the graphs")
    x = vectors[:,keep][:,:dimensions]*np.sqrt(values[keep][:dimensions])
    labels = (AgglomerativeClustering(n_clusters=groups).fit_predict(x) if method=="ward" else KMeans(n_clusters=groups,n_init=20,random_state=seed).fit_predict(x))
    diag = {"positive_dimensions": int(keep.sum()), "used_dimensions": x.shape[1],
            "negative_eigenvalue_fraction": float(np.abs(values[values<0]).sum()/np.abs(values).sum()),
            "method": method, "groups": groups, "seed": seed}
    return pd.DataFrame({"system": frame.index, "cluster": labels+1}), pd.DataFrame(x,index=frame.index), diag
