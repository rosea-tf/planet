# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
import numpy as np
from sklearn.decomposition import PCA
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split


# %%
filedata = []
os.chdir("results/statecorrel")
for filename in os.listdir("."):
    if filename.endswith(".npz"):
        filedata.append(np.load(filename))


# %%
qf_data = {}
for k in filedata[0].keys():
    qf_data[k] = np.concatenate([a[k] for a in filedata], axis=0)


# %%
hdim = {k: v.shape for k, v in qf_data.items()}
print(hdim)


# %%
# create a pca'd thing from combined deterministic & stochastic representations

qf_data['pca'] = PCA(n_components=(hdim['position'] + hdim['velocity'])).fit_transform(
    np.concatenate([qf_data['sample'], qf_data['belief']], axis=1))


# %%
# calc cov: each of the 4 combinations
qf_correl = {}
ols_coefs = {}
ols_scores = {}

for nx in ['position', 'velocity']:
    tx = qf_data[nx]
  
    for ny in ['sample', 'belief']:
        ty = qf_data[ny]

        xy_label = '{}{}'.format(nx[0], ny[0])

        # temporarily concatenate
        txy = np.concatenate([tx, ty], axis=1)

        #calculate correlation
        cov = np.abs(np.corrcoef(txy, rowvar=False))

        # cut out relevant part of corr matrix:
        # C[i,j] = cov(x[i], y[j])
        qf_correl[xy_label] = cov[:tx.shape[1], tx.shape[1]:]
        
        # look for disentangled relationships
        lr = LinearRegression()
        tx_train, tx_test, ty_train, ty_test = train_test_split(tx, ty) #0.25
        
        ols_coefs[xy_label] = np.empty([ty.shape[1], tx.shape[1]])
        ols_scores[xy_label] = np.empty([tx.shape[1]])
        
        for ix in range(tx.shape[1]):
            lr.fit(X=ty_train, y=tx_train[:, ix])
            ols_coefs[xy_label][:, ix] = lr.coef_
            ols_scores[xy_label][ix] = lr.score(X=ty_test, y=tx_test[:, ix])


# %%
lr = LinearRegression()
lr.fit(X=ty[1050:], y=tx[1050:, 0])
lr.score(X=ty[:1050], y=tx[:1050, 0])


# %%
lr.coef_.shape


# %%
for ix, nx in enumerate(['position', 'velocity']):
    for iy, ny in enumerate(['sample', 'belief']):
        xy_label = '{}{}'.format(nx[0], ny[0])
        print (qf_correl[xy_label].shape)


# %%
qf_data['sample']


# %%
# fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(10,6), sharey='row', sharex='col')#, sharex='col')#, sharey=True)#, sharex=True, sharey=True)

fig = plt.figure(figsize=(10,4), constrained_layout=True)

# fig.suptitle ("Correlations between learned latent space and cheetah_run latents")
for iy, ny in enumerate(['belief', 'sample']):
    for ix, nx in enumerate(['position', 'velocity']):
        xy_label = '{}{}'.format(nx[0], ny[0])
        # axo = ax[iy + ix * 2]
        if ix > 0:
            sharey = fig.get_axes()[iy*2]
        else:
            sharey = None
        axo = fig.add_subplot(1, 4, ix + iy * 2 + 1, sharey=sharey)
        axo.set_title(r"$\rho$ ({}, {})".format(nx, ny))
        axo.set_xlabel("{} element".format(nx))
        if sharey is None:
            axo.set_ylabel("{} element".format(ny))
        axo.set_xticks([])
        axo.set_yticks([])
        im = axo.imshow(qf_correl[xy_label].T, vmin=0.1, vmax=0.9)
        axo.set_aspect('auto')
        ax[iy + ix * 2].autoscale(True)               

# fig.colorbar(im, ax=fig.get_axes(), fraction=0.046, pad=0.04, orientation="horizontal")
fig.colorbar(im, ax=fig.get_axes(), fraction=0.046, pad=0.04, orientation="vertical")
fig.tight_layout()
fig.subplots_adjust(right=0.825)
fig.savefig("statecorrel.png", dpi=150, bbox_inches='tight')


# %%
fig.get_axes()[0]


# %%
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(7,14), sharey='row', sharex='col')#, sharex='col')#, sharey=True)#, sharex=True, sharey=True)
fig.suptitle ("OLS weights between learned latent space and cheetah_run latents")
for ix, nx in enumerate(['position', 'velocity']):
    for iy, ny in enumerate(['belief', 'sample']):
        xy_label = '{}{}'.format(nx[0], ny[0])

        axo = ax[iy, ix]
        axo.set_title(r"$\rho$ ({}, {})".format(nx, ny))
        axo.set_xlabel("{} element".format(nx))
        axo.set_ylabel("{} element".format(ny))
        axo.set_xticks([])
        axo.set_yticks([])
        im = axo.imshow(ols_coefs[xy_label], vmin=-0.15, vmax=1)
        axo.set_aspect('auto')
        ax[iy, ix].autoscale(True)               

fig.colorbar(im, ax=ax.ravel().tolist(), fraction=0.046, pad=0.04, orientation="horizontal")
fig.tight_layout()
fig.subplots_adjust(top=0.925, bottom=0.2)
# fig.savefig("correl.png", dpi=150, bbox_inches='tight')


# %%
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(7,3), sharey='row', sharex='col')#, sharex='col')#, sharey=True)#, sharex=True, sharey=True)
fig.suptitle ("$R^2$ of OLS regression of learned latents onto cheetah_run latents")
for ix, nx in enumerate(['position', 'velocity']):
    for iy, ny in enumerate(['belief', 'sample']):
        xy_label = '{}{}'.format(nx[0], ny[0])

        axo = ax[iy, ix]
        axo.set_title(r"$\rho$ ({}, {})".format(nx, ny))
        axo.set_xlabel("{} element".format(nx))
#         axo.set_ylabel("{} element".format(ny))
        axo.set_xticks([])
        axo.set_yticks([])
        im = axo.imshow(ols_scores[xy_label][np.newaxis, :], vmin=0, vmax=1, )
#         axo.set_aspect('auto')
#         ax[iy, ix].autoscale(True)               

fig.colorbar(im, ax=ax.ravel().tolist(), fraction=0.046, pad=0.04, orientation="horizontal")
# fig.tight_layout()
fig.subplots_adjust(top=0.95, bottom=0.25)
# fig.savefig("correl.png", dpi=150, bbox_inches='tight')


# %%
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(qf_data['pca'], qf_data['position'])


# %%
reg.score(qf_data['pca'], qf_data['position'])

