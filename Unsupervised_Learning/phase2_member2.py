"""Run from the repository root with NumPy, pandas and Pillow."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

root = Path(__file__).resolve().parent
results = root / "clustering_results"
data = pd.read_csv(root.parent / "Dataset/preprocessed_data.csv")
features = json.loads((results / "kmeans_features.json").read_text())
expected = [c for c in data if c not in {"ID", "split", "default.payment.next.month"}
            and not c.startswith("EDUCATION_")]
assert features == expected
train = data.split.eq("train").to_numpy()
values = data[features].to_numpy()
x = values[train]
assert x.shape == (24000, 22) and values.shape == (30000, 22)
assert np.isfinite(values).all() and np.allclose(x.mean(0), 0, atol=.01)
assert np.allclose(x.std(0), 1, atol=.01)
sample = np.random.RandomState(42).choice(len(x), 5000, replace=False)

def distance(a, b):
    squared = (a**2).sum(1)[:, None] + (b**2).sum(1) - 2 * (a @ b.T)
    return np.sqrt(np.maximum(squared, 0))

pairwise = distance(x[sample], x[sample])
np.fill_diagonal(pairwise, 0)

def silhouette(labels):
    _, codes = np.unique(labels[sample], return_inverse=True)
    groups = np.eye(codes.max()+1)[codes]
    counts = groups.sum(0)
    if len(counts) < 2:
        return np.nan
    totals = pairwise @ groups
    rows = np.arange(len(codes))
    within = totals[rows, codes] / np.maximum(counts[codes]-1, 1)
    averages = totals / counts
    averages[rows, codes] = np.inf
    nearest = averages.min(1)
    denom = np.maximum(within, nearest)
    scores = np.divide(nearest-within, denom, out=np.zeros_like(within), where=denom>0)
    return np.where(counts[codes]==1, 0, scores).mean()

def davies_bouldin(labels):
    groups = [x[labels==c] for c in np.unique(labels)]
    centers = np.array([g.mean(0) for g in groups])
    spread = np.array([np.linalg.norm(g-c, axis=1).mean() for g,c in zip(groups,centers)])
    separation = distance(centers, centers)
    np.fill_diagonal(separation, np.inf)
    return ((spread[:,None]+spread)/separation).max(1).mean()

saved = pd.read_csv(results / "kmeans_labels.csv").set_index(["ID","split"])
clients = pd.MultiIndex.from_frame(data[["ID","split"]])
assert saved.index.is_unique and clients.is_unique and len(saved)==len(clients)
baseline = saved.reindex(clients).kmeans_cluster
assert baseline.notna().all()
km = baseline.to_numpy()[train]
reference = pd.read_csv(results / "kmeans_final_metrics.csv").iloc[0]
assert int(reference["k"]) == len(np.unique(km))
assert np.isclose(silhouette(km), reference["Silhouette (5,000 sample)"])
assert np.isclose(davies_bouldin(km), reference["Davies-Bouldin"])

def posterior(rows, weights, means, variances):
    logs = np.column_stack([
        np.log(w)-.5*(np.log(2*np.pi*v).sum()+((rows-m)**2/v).sum(1))
        for w,m,v in zip(weights,means,variances)
    ])
    peak = logs.max(1, keepdims=True)
    shifted = np.exp(logs-peak)
    total = shifted.sum(1, keepdims=True)
    return shifted/total, (peak+np.log(total)).sum()

def fit(k, covariance, seed):
    means = x[np.random.default_rng(seed).choice(len(x),k,replace=False)]
    weights = np.full(k,1/k)
    variances = np.tile(x.var(0),(k,1))
    previous = -np.inf
    squared = x**2
    for step in range(1,201):
        model = weights,means,variances
        prob, likelihood = posterior(x,*model)
        if abs(likelihood-previous)/len(x)<1e-4:
            return model, likelihood, step, True
        mass = prob.sum(0)
        if (mass<1e-12).any():
            break
        previous = likelihood
        weights = mass/len(x)
        means = prob.T @ x / mass[:,None]
        variances = np.maximum(prob.T @ squared / mass[:,None] - means**2,0)
        if covariance=="spherical":
            variances = np.broadcast_to(variances.mean(1,keepdims=True),means.shape)
        variances = np.maximum(variances, .001)
    return model, likelihood, step, False

rows, models = [], []
for k in range(2,7):
    for covariance in ("diagonal","spherical"):
        fits = [fit(k,covariance,seed) for seed in (42,43)]
        converged = [f for f in fits if f[3]]
        model, likelihood, steps, success = max(converged or fits, key=lambda f:f[1])
        labels = posterior(x,*model)[0].argmax(1)
        rows.append(dict(components=k,covariance=covariance,converged=success,
                         iterations=steps,smallest_cluster=np.bincount(labels,minlength=k).min(),
                         silhouette=silhouette(labels),log_likelihood=likelihood))
        models.append(model)
search = pd.DataFrame(rows)
eligible = search[search.converged & search.smallest_cluster.gt(0) & search.silhouette.notna()]
best = eligible.sort_values("silhouette",ascending=False,kind="stable").iloc[0]
model = models[best.name]
probabilities = posterior(values,*model)[0]
gmm_all = probabilities.argmax(1)
gmm = gmm_all[train]
comparison = pd.DataFrame([
    dict(algorithm=name,clusters=len(np.unique(labels)),
         silhouette=silhouette(labels),davies_bouldin=davies_bouldin(labels))
    for name,labels in [("K-Means",km),("Gaussian Mixture",gmm)]
])
assignments = data[["ID","split"]].assign(gmm_cluster=gmm_all,gmm_membership=probabilities.max(1))
sizes = assignments.groupby(["gmm_cluster","split"]).size().unstack(fill_value=0)
overlap = pd.crosstab(km,gmm,rownames=["kmeans_cluster"],colnames=["gmm_cluster"])
search.assign(selected=search.index==best.name).to_csv(results/"gmm_parameter_search.csv",index=False)
comparison.to_csv(results/"gmm_kmeans_comparison.csv",index=False)
assignments.to_csv(results/"gmm_labels.csv",index=False)
sizes.to_csv(results/"gmm_cluster_sizes.csv")
overlap.to_csv(results/"gmm_kmeans_train_overlap.csv")
models_dir = root/"models"
models_dir.mkdir(exist_ok=True)
np.savez_compressed(models_dir/"gmm_model.npz",weights=model[0],means=model[1],
                    variances=model[2],features=np.array(features))

# Fit common PCA axes on training rows for display only.
_, singular, axes = np.linalg.svd(x-x.mean(0),full_matrices=False)
projected = (x[sample]-x.mean(0)) @ axes[:2].T
variance = singular[:2]**2 / (singular**2).sum()
coords = (projected-projected.min(0)) / np.ptp(projected,axis=0)
colors = ["#2455A4","#B94322","#138567","#8647B1","#B47813","#287984"]
image = Image.new("RGB",(1200,620),"white")
draw = ImageDraw.Draw(image)
draw.text((40,22),"Cluster assignments on the same 5,000 training clients",fill="#111827",font_size=26)
for offset,name,labels in [(45,"K-Means",km),(635,"Gaussian Mixture",gmm)]:
    draw.text((offset,72),name,fill="#111827",font_size=22)
    draw.rectangle((offset,110,offset+520,520),outline="#68768A")
    for (a,b),label in zip(coords,labels[sample]):
        px,py = offset+5+a*510,515-b*400
        draw.ellipse((px-2,py-2,px+2,py+2),fill=colors[label])
    draw.text((offset,535),f"Horizontal PC1 ({variance[0]:.1%}) | Vertical PC2 ({variance[1]:.1%})",
              fill="#29344A",font_size=17)
    for i,label in enumerate(np.unique(labels)):
        left = offset+i*78
        draw.rectangle((left,576,left+12,588),fill=colors[label])
        draw.text((left+17,572),str(label),fill="#111827",font_size=17)
plots = root/"plots"
plots.mkdir(exist_ok=True)
image.save(plots/"gmm_kmeans_pca.png")
print(best.to_string(),comparison.to_string(index=False),sizes.to_string(),overlap.to_string(),sep="\n\n")
