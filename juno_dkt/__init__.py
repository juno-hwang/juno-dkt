import numpy as np
import pandas as pd
import networkx as nx
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from juno_dkt.utils import *

eps = 1e-8

class ItemEncoder:
	def __init__(self, n_items=None, binary_only=False):
		self.n_items = n_items
		self.binary_only = binary_only

	def transform(self, students, items, answers):
		if self.binary_only:
			idx = np.floor(answers) == answers
			students = np.array(students[idx])
			items = np.array(items[idx])
			answers = np.array(answers[idx])
		print('[1/3] One-hot encoding item responces...')
		encoded, idxs = self.one_hot_encode(items, answers)
		batches = self.batchify(students, encoded, idxs)
		print('Number of batches(students) : ', len(batches))
		return batches

	def one_hot_encode(self, items, answers):
		if self.n_items == None :
			self.n_items = len(list(set(items)))

		self.item_ids = list(set(items))
		item_id_to_idx = {}
		for iid in self.item_ids:
			item_id_to_idx[iid] = self.item_ids.index(iid)

		encoded = np.zeros((len(items), self.n_items*2))
		idxs = []
		for i in tqdm(range(len(items))):
			idx = item_id_to_idx[items[i]]
			encoded[i, idx] += answers[i]
			encoded[i, idx + self.n_items] += 1 - answers[i]
			idxs.append(idx)

		return encoded, idxs

	def inverse_transform(self, encoded):
		if self.n_items == None:
			print('[Warning] Items are not assigned. The method ItemEncoder.transform(items, answers) must be called before.')
			raise ValueError
		else:
			# 추후에 추가해야함
			pass

	def idx_to_item(self, idx):
		# 추가해야함
		pass

	def batchify(self, student_id, encoded, idxs):
		student_set = list(set(student_id))
		student_id_to_idx = {}
		for sid in student_set:
			student_id_to_idx[sid] = student_set.index(sid)

		batches = [ [] for i in range(len(student_set)) ]
		idxs_batches = [ [] for i in range(len(student_set)) ]
		transition_count = np.zeros((self.n_items, self.n_items))

		print('[2/3] Batchifying one-hot vectors...')
		for i in tqdm(range(len(student_id))) :
			student_idx = student_id_to_idx[student_id[i]]
			batches[student_idx].append( encoded[i] )
			idxs_batches[student_idx].append( idxs[i] )
			if len(idxs_batches[student_idx]) > 1:
				i = idxs_batches[student_idx][-2]
				j = idxs_batches[student_idx][-1]
				if i != j:
					transition_count[i,j] += 1

		print('[3/3] Converting type into torch.Tensor...')
		for i, b in enumerate(tqdm(batches)):
			batches[i] = torch.Tensor(b).view(-1,2*self.n_items)
		self.transition_count = transition_count

		return batches

class DKT(nn.Module):
	def __init__(self, n_hidden, batch_size, lr, n_embedding=None, device='cpu'):
		super().__init__()
		self.n_hidden = n_hidden
		self.lr = lr
		self.batch_size = batch_size
		self.n_embedding = n_embedding
		self.device = torch.device(device)
		self.to(device)

	def forward(self, x):
		h, c = self.lstm(x)
		y_ = self.decoder(h)
		return y_

	def fit(self, batches, n_iter, test_set=None):
		try :
			self.n_items
		except :
			self.n_items = batches[0].shape[-1]//2
			if self.n_embedding == None:
				self.lstm = nn.LSTM(batches[0].shape[-1], self.n_hidden)
			else:
				self.lstm = nn.Sequential(nn.Linear(batches[0].shape[-1], self.n_embedding),
										  nn.LSTM(self.n_embedding, self.n_hidden))
			self.decoder = nn.Sequential(nn.Linear(self.n_hidden, self.n_items),
										 nn.Dropout(),
										 nn.Sigmoid())
			self.to(self.device)

		self.opt = optim.Adam(self.parameters(), lr=self.lr)
		loader = DataLoader(batches, shuffle=True, batch_size=self.batch_size, collate_fn=collate)

		for n in range(n_iter):
			print('=== Training epoch %d ==='%(n+1))
			iteration = tqdm(loader)
			self.train()
			loss_history = []
			for i, data in enumerate(iteration):
				data = data.to(self.device)
				if data.shape[0] < 3 :
					continue
				y_ = self(data)

				self.opt.zero_grad()
				loss = self._loss(y_[:-1], data[1:])
				if torch.isnan(loss) :
					print('\n===========NaN detected!!!==============\n')
					print(i)
					continue

				loss.backward()
				self.opt.step()
				loss_history.append(loss.item())
				iteration.set_description_str('loss : %.4f'%loss_history[-1] if len(loss_history) else '')

			if test_set != None:
				print('Test score : ',
					  'ROC AUC %.5f'%self.roc_auc_score(test_set),
					  ' / Binary Cross Entropy %.5f'%self.bce_score(test_set))
		self.eval()

	def _loss(self, data, target):
		delta = target[:,:, :self.n_items] + target[:,:, self.n_items:]
		mask = delta.sum(axis=-1).type(torch.BoolTensor).to(self.device)
		data =  data* (1-2*eps) + eps
		correct = target[:,:, :self.n_items].to(self.device)
		bce = - correct*data.log() - (1-correct)*(1-data).log()
		bce = (bce*delta).sum(axis=-1)
		return torch.masked_select(bce, mask).mean()

	def y_true_and_score(self, batches):
		self.eval()
		loader = DataLoader(batches, batch_size=64, collate_fn=collate)
		y_true, y_score = [], []

		with torch.no_grad():
			for i, data in enumerate(loader):
				data = data.to(self.device)
				correct = data[:,:, :self.n_items].to(self.device)
				mask = (data[:,:, :self.n_items] + data[:,:, self.n_items:]).type(torch.BoolTensor).to(self.device)

				y_ = self(data)
				y_true.append( torch.masked_select(correct[1:], mask[1:]) )
				y_score.append( torch.masked_select(y_[:-1], mask[1:]) )

		y_true = torch.cat(y_true).detach().cpu().numpy()
		y_score = torch.cat(y_score).detach().cpu().numpy()
		return y_true, y_score

	def predict(self, data):
		self.eval()
		if type(data) == list:
			loader = DataLoader(data, batch_size=64, collate_fn=collate, shuffle=False)
			predictions = []

			with torch.no_grad():
				for i, data in enumerate(loader):
					data = data.to(self.device)
					delta = data[:,:, :self.n_items] + data[:,:, self.n_items:]
					length = delta.sum(axis=-1).sum(axis=0).type(torch.IntTensor)

					y_ = self(data)
					for i in range(y_.shape[1]):
						predictions.append( y_[:length[i], i, :].detach().cpu().numpy() )

			return predictions

		elif type(data) == torch.Tensor:
			y_ = self(torch.unsqueeze(data.to(self.device), 1))
			return y_.detach().cpu().numpy()[:,0,:]

	def roc_auc_score(self, batches):
		# Note : The score is evaluated on binary y_true items only.
		y_true, y_score = self.y_true_and_score(batches)
		idx = np.floor(y_true) == y_true
		return roc_auc_score(y_true[idx], y_score[idx])

	def bce_score(self, batches):
		# Note : The score is evaluated on binary y_true items only.
		y_true, y_score = self.y_true_and_score(batches)
		bce = y_true*np.log(y_score) + (1-y_true)*np.log(1-y_score)
		return -np.mean(bce)

	def influence_matrix(self):
		matrix = self._conditional_predict_matrix()
		matrix = matrix / np.sum(matrix, axis=0)
		return matrix

	def _conditional_predict_matrix(self):
		self.eval()
		x = torch.zeros(size=(1, self.n_items, self.n_items*2)).to(self.device)
		for n in range(self.n_items):
			x[0, n, n] = 1

		with torch.no_grad():
			return self(x)[0].detach().cpu().numpy()

	def graph(self, item_encoder, method='conditional', use_label=False, threshold=0.1, pair_threshold=0):
		count = item_encoder.transition_count
		mask = count > pair_threshold

		if method == 'transition':
			mat = item_encoder.transition_count.T
			mat /= np.sum(mat, axis=0)
			mat = mat.T * mask
		elif method == 'conditional':
			cond = self._conditional_predict_matrix()
			mat = item_encoder.transition_count.T
			mat /= np.sum(mat, axis=0)
			mat = mat.T * cond * mask
		else:
			print('[Warning] Wrong method entered.')
			return

		edges = []
		for i in range(self.n_items):
			for j in range(self.n_items):
				if i!=j and mat[i,j] >= threshold and mat[i,j] > mat[j,i]:
					if use_label:
						edges.append((item_encoder.item_ids[i], item_encoder.item_ids[j], mat[i,j]))
					else:
						edges.append((i,j, mat[i,j]))

		print(np.mean(mat), np.std(mat), np.max(mat))

		g = nx.DiGraph()
		if use_label:
			g.add_nodes_from(item_encoder.item_ids)
		else :
			g.add_nodes_from(range(self.n_items))
		g.add_weighted_edges_from(edges)
		return g