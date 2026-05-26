import warnings
warnings.filterwarnings("ignore")

import episcanpy as epi
import scanpy as sc
import pandas as pd
import numpy as np
import sklearn
import anndata
import scipy
import gc


colors_to_use=[(0.34550725069638827, 0.4203708006658883, 0.9696902293486781),
 (0.9893800026041992, 0.378955911742755, 0.21756841368122667),
 (0.3959642074605608, 0.24823947872676938, 0.4016676539297192),
 (0.9937826924994482, 0.4211527500079969, 0.8812994030921271),
 (0.4140058397372807, 0.9619317608252869, 0.3109026417629064),
 (0.2286247431221609, 0.6437632542888629, 0.4081322805120583),
 (0.25003260615661993, 0.938691496932296, 0.9192515923797947),
 (0.7646511697684856, 0.24254983894398235, 0.7085129830496552),
 (0.3017721221187747, 0.6522700618245787, 0.9844707721904342),
 (0.21782892631529166, 0.2854088109905996, 0.7174819557293214),
 (0.9991896546476063, 0.4844986464266022, 0.5344476773522967),
 (0.8919236560192338, 0.7949117963224906, 0.7730486745511909),
 (0.6428145648345002, 0.31108252586505475, 0.2041098261347507),
 (0.20269999368314223, 0.7748472379028853, 0.6824940025160084),
 (0.9819923318374866, 0.6965816490867496, 0.21263663000131983),
 (0.809020033101876, 0.23056728504993104, 0.9541856467035792),
 (0.9191255237720919, 0.20589563833687236, 0.4656154484972612),
 (0.4066816185888487, 0.7781338620666193, 0.22302116197384453),
 (0.9393586762460103, 0.9614694589148117, 0.22619331869433382),
 (0.29226458048518345, 0.42830335364029093, 0.6916335500700534)]
	
def create_count_matrix(fragments_file : str, valid_bcs : list, features_space : str, features_file=None, source=None, metadata=None):

	if not isinstance(fragments_file, str) or not isinstance(valid_bcs, list) or not isinstance(features_space, str):
		print(type(fragments_file), type(valid_bcs), type(features_space), flush=True)
		raise TypeError
	
	if features_space == "Window":
		adata = epi.ct.window_mtx(fragments_file=fragments_file, valid_bcs=valid_bcs, fast=False)
		
	elif features_space != "Window":
		if features_file[-6:]=="gtf.gz" or features_file[-3:]=="gtf":
			if features_space=="GA":
				adata = epi.ct.gene_activity_mtx(fragments_file=fragments_file, gtf_file=features_file, valid_bcs=valid_bcs, fast=False)
			else:
				data={}
				data["tRNA"]=[30,80] #POL3
				data["rRNA"]=[30,80] #POL1/POL3
				data["protein_coding"]=[5000,2000] #POL2
				data["lncRNA"]=[30,80] #POL3
				data["miRNA"]=[5000,2000] #POl2
				data["telomerase_RNA"]=[5000,2000] #POL2
				adata = epi.ct.gene_activity_mtx(fragments_file=fragments_file, gtf_file=features_file, valid_bcs=valid_bcs, 
								 upstream=data[features_space][0], downstream=data[features_space][1], 
								 source=source, gene_type=[features_space], fast=False)
								 	
		elif features_file[-10:]=="narrowPeak" or features_file[-3:]=="bed":
			adata = epi.ct.peak_mtx(fragments_file=fragments_file, peak_file=features_file, valid_bcs=valid_bcs, normalized_peak_size=None, fast=False)
			adata.var.index=["_".join([str(adata.var.iloc[i][0]),str(adata.var.iloc[i][1]),str(adata.var.iloc[i][2])]) for i in range(len(adata.var.index))]
	
	adata.X=scipy.sparse.csr_matrix(adata.X, dtype="float32")
	adata.var.columns = adata.var.columns.astype(str)
	adata.obs.columns = adata.obs.columns.astype(str)
	adata.var = adata.var.rename(columns={"0": "chr", "1": "start", "2" : "stop"})
	adata.var_names_make_unique(join="_")
	adata = adata[:,adata.var[adata.var['chr'].str.match('chr')].index]
	adata = adata[:,adata.var[~adata.var['chr'].str.match('chrM')].index]
	if features_file[-10:]=="narrowPeak" or features_file[-3:]=="bed" or features_space == "Window":
		a = [a for a in adata.var.index if len(a.split("_"))!=3]
		adata = adata[ :, adata.var.loc[list(set(adata.var.index) - set(a))].index ]
		del a
	adata.obs = metadata.loc[adata.obs.index]

	return adata
  
def qc_filtering(adata, omic="ATAC"):
	epi.pp.qc_stats(adata, verbose=False)
		
	if omic=="GEX":
		mt_gene_mask = np.flatnonzero([gene.startswith(("MT-", "mt-")) for gene in adata.var_names])
		adata.obs['mt_frac'] = np.sum(adata[:, mt_gene_mask].X, axis=1).A1/adata.obs['n_counts']
		ribo_gene_mask = np.flatnonzero([gene.startswith(("Rps", "Rpl")) for gene in adata.var_names])
		adata.obs['ribofrac'] = np.sum(adata[:, ribo_gene_mask].X, axis=1).A1/adata.obs['n_counts']
		
		
	min_features = 10**np.quantile(adata.obs["log_n_features"], 0.05)
	max_features = 10**np.quantile(adata.obs["log_n_features"], 0.95)
	min_cells = 10**np.quantile(adata.var["log_n_cells"], 0.05)
	max_cells = 10**np.quantile(adata.var["log_n_cells"], 0.95)

	print("Adata's shape:", adata.shape, flush=True)
	
	epi.pp.set_filter(adata, "n_features", min_threshold=min_features, max_threshold=max_features, verbose=False)
	epi.pp.set_filter(adata, "n_cells", min_threshold=min_cells, max_threshold=max_cells, verbose=False)
	adata = epi.pp.apply_filters(adata, verbose=False)
	print("Adata's shape after cells and features filtering:", adata.shape, flush=True)
	
	if omic=="GEX":
		adata = adata[adata.obs["mt_frac"]<=0.05]
		print("Adata's shape after MT filtering:", adata.shape, flush=True)
		
		sc.pp.scrublet(adata)
		adata=adata[adata.obs.predicted_doublet==False]
		print("Adata's shape after doublets filtering:", adata.shape, flush=True)
   
	sc.pp.highly_variable_genes(adata, flavor='seurat_v3')

	adata.layers["counts"]=adata.X.copy()

	if omic == "GEX":
		sc.pp.normalize_total(adata)
		adata.X=np.sqrt(adata.X)
	else:
		epi.pp.tfidf(adata)
		adata.X=adata.layers["tf-idf"]
		del adata.layers["tf-idf"]

	print("Adata's shape after HVG filtering:", adata.shape, flush=True)
	
	

	   
	return adata 
 
			
def intersection(l):
	return np.array(list(set.intersection(*map(set,list(l)))))


def flat_list(l):
	return np.array(list(set([item for sublist in l for item in sublist])))


def most_common(l):
	l=list(l)
	return max(set(l), key=l.count)

	
def my_polynom(params, x):
	n = len(params)
	assert n > 0
	x_vec = np.array([np.power(x, i) for i in range(n - 1, -1, -1)])
	return x_vec.T @ np.array(params)


def procrustes(X, Y, scaling=True, reflection='best'):

	"""
	Taken from https://stackoverrun.com/es/q/5162566 adaptation of MATLAB to numpy
	"""

	"""
	A port of MATLAB's `procrustes` function to Numpy.

	Procrustes analysis determines a linear transformation (translation,
	reflection, orthogonal rotation and scaling) of the points in Y to best
	conform them to the points in matrix X, using the sum of squared errors
	as the goodness of fit criterion.

		d, Z, [tform] = procrustes(X, Y)

	Inputs:
	------------
	X, Y	
		matrices of target and input coordinates. they must have equal
		numbers of  points (rows), but Y may have fewer dimensions
		(columns) than X.

	scaling 
		if False, the scaling component of the transformation is forced
		to 1

	reflection
		if 'best' (default), the transformation solution may or may not
		include a reflection component, depending on which fits the data
		best. setting reflection to True or False forces a solution with
		reflection or no reflection respectively.

	Outputs
	------------
	d	   
		the residual sum of squared errors, normalized according to a
		measure of the scale of X, ((X - X.mean(0))**2).sum()

	Z
		the matrix of transformed Y-values

	tform   
		a dict specifying the rotation, translation and scaling that
		maps X --> Y

	"""

	n,m = X.shape
	ny,my = Y.shape

	muX = X.mean(0)
	muY = Y.mean(0)

	X0 = X - muX
	Y0 = Y - muY

	ssX = (X0**2.).sum()
	ssY = (Y0**2.).sum()

	# centred Frobenius norm
	normX = np.sqrt(ssX)
	normY = np.sqrt(ssY)

	# scale to equal (unit) norm
	X0 /= normX
	Y0 /= normY

	if my < m:
		Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

	# optimum rotation matrix of Y
	A = np.dot(X0.T, Y0)
	U,s,Vt = np.linalg.svd(A,full_matrices=False)
	V = Vt.T
	T = np.dot(V, U.T)

	if reflection != 'best':

		# does the current solution use a reflection?
		have_reflection = np.linalg.det(T) < 0

		# if that's not what was specified, force another reflection
		if reflection != have_reflection:
			V[:,-1] *= -1
			s[-1] *= -1
			T = np.dot(V, U.T)

	traceTA = s.sum()

	if scaling:

		# optimum scaling of Y
		b = traceTA * normX / normY

		# standarised distance between X and b*Y*T + c
		d = 1 - traceTA**2

		# transformed coords
		Z = normX*traceTA*np.dot(Y0, T) + muX

	else:
		b = 1
		d = 1 + ssY/ssX - 2 * traceTA * normY / normX
		Z = normY*np.dot(Y0, T) + muX

	# transformation matrix
	if my < m:
		T = T[:my,:]
	c = muX - b*np.dot(muY, T)

	#transformation values
	tform = {'rotation':T, 'scale':b, 'translation':c}

	return d, Z, tform


# Convert input to dense float32 arrays, handling sparse inputs
def to_dense_float32(X):
	return X.toarray().astype(np.float32) if scipy.sparse.issparse(X) else X.astype(np.float32)
