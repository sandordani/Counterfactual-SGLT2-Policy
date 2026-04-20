from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mlp_classifier import MLPClassifier, train_mlp, predict_mlp
from sklearn.utils.class_weight import compute_class_weight

HIDDEN_DIMS = [128, 64]

lab_thresholds = {
    'CHOL-C': ('<', 5.2),
    'HDL-C': ('>', 1.2),
    'LDL-C_D': ('<', 3.4),
    'GLU-L': ('-', 3.6, 6.0),
    'HbA1C': ('-', 4.2, 6.1),
    'GPT-L': ('<', 40),
    'GGT-L': ('-', 7, 50),
    'GOT-L': ('<', 40),
    'KREA-L': ('-', 62, 106),
    'EGFR': ('<', 60),
    'CRP-L': ('<', 5.2)
}

rename_diseases = {
    'hepatopathia': 'Hepatopathia',
    'zsírmáj': 'Fatty liver',
    'diabetes': 'Diabetes',
    'cardiovascular disease': 'Cardiovascular Disease',
    'cererovascular disease': 'Cererovascular Disease',
    'periferial arterial disease': 'Periferial Arterial Disease',
    'thrombosis': 'Thrombosis',
    'renal diseases': 'Renal Diseases',
    'thyeroid diseases': 'Thyeroid Diseases',
    'atherosclerosis': 'Atherosclerosis',
    'lipid metabolism disorders': 'Lipid Metabolism Disorders'
}

labs = list(lab_thresholds.keys())

observation_cols = labs + ['PRESSURE_CAT', 'BMI_CAT', 'ESETKOR', 'NEM']
treatment_col = 'TIPUS'
diag_cols = list(rename_diseases.values())

device = 'cuda:0'

def truncated_gumbel(logit, truncation):
    assert not torch.isneginf(logit).any()

    gumbel = torch.tensor(np.random.gumbel(size=truncation.shape), dtype=torch.float32).to(device) + logit
    trunc_g = -torch.log(torch.exp(-gumbel) + torch.exp(-truncation))
    return trunc_g

def topdown(logits, x_stars, nsamp=1):
    torch.testing.assert_close(torch.sum(torch.exp(logits), 1), torch.tensor([1 for _ in range(logits.shape[0])], dtype=torch.float32).to(device)), "Probabilities do not sum to 1"
    ncat = logits.shape[1]
    
    gumbels = torch.zeros((nsamp, logits.shape[0], logits.shape[1]), dtype=torch.float32).to(device)

    # Sample top gumbels
    topgumbel = torch.tensor(np.random.gumbel(size=(nsamp, logits.shape[0])), dtype=torch.float32).to(device)

    for i in range(ncat):
        mask_obs = (i == x_stars)
        mask_poss = (i != x_stars) & (~torch.isneginf(logits[:, i]))
        mask_zero = torch.isneginf(logits[:, i])
        # This is the observed outcome
        gumbels[:, mask_obs, i] = topgumbel[:, mask_obs] - logits[mask_obs, i]
        # These were the other feasible options (p > 0)
        gumbels[:, mask_poss, i] = truncated_gumbel(logits[mask_poss, i], topgumbel[:, mask_poss]) - logits[mask_poss, i]
        # These have zero probability to start with, so are unconstrained
        gumbels[:, mask_zero, i] = torch.tensor(np.random.gumbel(size=(nsamp, torch.sum(mask_zero).item())), dtype=torch.float32).to(device)
    return gumbels


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = bce * ((1 - p_t) ** self.gamma)
        return loss.mean()
    

def get_followups(data, next_esetazons):
    df = data.set_index("ESETAZON")
    return df.loc[next_esetazons]

class SCM():
    def __init__(self, data, obs_train_epochs=10, diag_train_epochs=10, obs_variable_values=None):
        self.data = data
        self.models = {}
        self.encoders = {}
        if obs_variable_values is None:
            self.n_classes = {obs_col: data[obs_col].nunique() for obs_col in observation_cols}
        else:
            self.n_classes = {obs_col: len(obs_variable_values[i]) for i, obs_col in enumerate(observation_cols)}

        # fit next obs only on cases with followup
        cases_with_followup = data['Next ESETAZON'].notna() & data['Next ESETAZON'].isin(data['ESETAZON']) # Megcserélni?
        followups = get_followups(data, data.loc[cases_with_followup, 'Next ESETAZON'])

        # find encode and tensor obs_k
        obs_k = data.loc[cases_with_followup, observation_cols]
        # obs_k = self.encode_data(obs_k, 'obs', exclude_cols=['ESETKOR'], fit=True)
        obs_k = self.encode_data(obs_k, 'obs',  fit=True, categories=obs_variable_values)

        diag_k = data.loc[cases_with_followup, diag_cols]
        diag_k =  torch.tensor(diag_k.astype(np.float32).to_numpy(), dtype=torch.float32, device=device)
        
        # find encode and tensor treatment
        treatment_k = data.loc[cases_with_followup, [treatment_col]]
        treatment_k = self.encode_data(treatment_k, 'treatment', fit=True) # category definition might be needed later

        obs_k_next = followups.loc[:, observation_cols]
        
        self.fit_obs(obs_k, diag_k, treatment_k, obs_k_next, epochs=obs_train_epochs)

        # fit diag and treatment selection on everything
        obs_k = data.loc[:, observation_cols]
        # obs_k =  self.encode_data(obs_k, 'obs', exclude_cols=['ESETKOR'], fit=False)
        obs_k =  self.encode_data(obs_k, 'obs', fit=False)

        diag_k = data.loc[:, diag_cols]

        self.fit_diag(obs_k, diag_k, epochs=diag_train_epochs)

    def encode_data(self, data, encoder_name, exclude_cols=[], fit=False, categories=None):
        if fit:
            if categories is None:
                # Automatically determine categories from the data, should be avoided
                categories = [sorted(data[c].dropna().unique().tolist()) for c in  data.columns if c not in exclude_cols]
            self.encoders[encoder_name] = OneHotEncoder(handle_unknown='ignore', categories=categories)
            encoded_data = self.encoders[encoder_name].fit_transform(data.drop(exclude_cols, axis=1))
        else:
            encoded_data = self.encoders[encoder_name].transform(data.drop(exclude_cols, axis=1))
        data_tensor = torch.tensor(encoded_data.toarray(), dtype=torch.float32, device=device)
        excluded = torch.tensor(data[exclude_cols].values.reshape(-1, 1), dtype=torch.float32, device=device)
        if exclude_cols:
            data_tensor = torch.cat([data_tensor, excluded], dim=1)
            
            return data_tensor
        else:
            return data_tensor

    def get_column_from_tensor(self, tensor, encoder_name, original_col, exclude_cols=[]):
        encoder = self.encoders[encoder_name]
        input_cols = encoder.get_feature_names_out()
        
        indices = np.where([f.startswith(f"{original_col}_") for f in input_cols])[0]

        one_hot_part = tensor[:, :-len(exclude_cols)] if exclude_cols else tensor
        column_ohe = one_hot_part[:, indices]
        
        return column_ohe

    # fit obs_k+1 = f(obs_k, treatment_k)
    def fit_obs(self, obs_k, diag_k, treatment_k, obs_k_next, epochs):
        X = torch.cat([obs_k, diag_k, treatment_k], dim=1)
        
        self.models['obs_models'] = {}
        for obs_col in observation_cols:
            print(f'Fitting model for {obs_col}')
            if obs_col == 'NEM': # deterministic relationship
                self.models['obs_models'][obs_col] = None
                continue

            X_train = X[torch.tensor(obs_k_next[obs_col].notna().values, device=device)]


            y = obs_k_next.loc[obs_k_next[obs_col].notna(), obs_col].cat.codes
            clf = MLPClassifier(X_train.shape[1], self.n_classes[obs_col], hidden_dims=HIDDEN_DIMS).to(device)
            y = torch.tensor(y.array, dtype=torch.long, device=device)

            loss = nn.CrossEntropyLoss()
            data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
                X_train, y), batch_size=64, shuffle=True)
            
            train_mlp(clf, data_loader, criterion=loss, num_epochs=epochs, lr=1e-3) # 30 epoch
            self.models['obs_models'][obs_col] = clf       
            

    def evaluate_obs(self, obs_k, diag_k, treatment_k, obs_k_next):
        X = torch.cat([obs_k, diag_k, treatment_k], dim=1)

        for obs_col, clf in self.models['obs_models'].items():
            if clf is None:  # deterministic relationship
                # preds = self.get_column_from_tensor(obs_k, 'obs', obs_col, exclude_cols=['ESETKOR'])
                preds = self.get_column_from_tensor(obs_k, 'obs', obs_col)
                print(obs_col, 'acuracy', accuracy_score(obs_k_next.loc[obs_k_next[obs_col].notna(), obs_col].cat.codes, preds.argmax(axis=1).cpu().numpy().astype(int)))
                continue
            clf.eval()
            X_test = X[torch.tensor(obs_k_next[obs_col].notna().values, device=device)]

            y = obs_k_next.loc[obs_k_next[obs_col].notna(), obs_col].cat.codes
            y = torch.tensor(y.array, dtype=torch.long, device=device)
            data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
                X_test, y), batch_size=64, shuffle=False)
            preds = predict_mlp(clf, data_loader)
            print(obs_col, 'acuracy', accuracy_score(obs_k_next.loc[obs_k_next[obs_col].notna(), obs_col].cat.codes, preds.argmax(axis=1).cpu().numpy().astype(int)))


    def fit_diag(self, obs_k, diag_k, epochs):
        X = obs_k
        self.models['diag_models'] = {}

        for diag_col in diag_cols:
            print(f'Fitting model for {diag_col}')
            X_train = X[torch.tensor(diag_k[diag_col].notna().values, device=device)]

            y = diag_k.loc[diag_k[diag_col].notna(), diag_col].cat.codes
            y = torch.tensor(y.array.reshape((-1, 1)), dtype=torch.float32, device=device)
            clf = MLPClassifier(X_train.shape[1], 1, hidden_dims=HIDDEN_DIMS).to(device)
            data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
                X_train, y), batch_size=64, shuffle=True)
            loss = FocalLoss(gamma=2.0)

            train_mlp(clf, data_loader, criterion=loss, num_epochs=epochs, lr=1e-3)  # 30 epoch
            self.models['diag_models'][diag_col] = clf

    def evaluate_diag(self, obs_k, diag_k):
            X = obs_k

            for diag_col, clf in self.models['diag_models'].items():
                X_test = X[torch.tensor(diag_k[diag_col].notna().values, device=device)]
                y = diag_k.loc[diag_k[diag_col].notna(), diag_col].cat.codes
                y = torch.tensor(y.array.reshape((-1, 1)), dtype=torch.float32, device=device)
                data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
                    X_test, y), batch_size=64, shuffle=False)
                clf.eval()
                preds = predict_mlp(clf, data_loader)
                print(diag_col, 'AUC', roc_auc_score(diag_k[diag_col], preds.cpu().numpy()))
                preds = torch.sigmoid(preds).cpu().numpy()
                preds = (preds > 0.5).astype(int)
                print(diag_col, 'accuracy:', accuracy_score(diag_k[diag_col], preds))
        
    def predict_obs(self, obs_k, diag_k, treatment_k, return_logits=False):
        X = torch.cat([obs_k, diag_k, treatment_k], dim=1)

        obs_preds = {}
        for obs_col, model in self.models['obs_models'].items():
            if model is None: 
                preds = self.get_column_from_tensor(obs_k, 'obs', obs_col)
            else:
                model.eval()
                logits = model(X)
                preds = torch.softmax(logits, dim=1)
            obs_preds[obs_col] = preds

        if return_logits:
            return obs_preds
        else:
            return torch.cat(list(obs_preds.values()), dim=1)
    
    def predict_diag(self, obs_k, probs=False):
        X = obs_k

        diag_probs = []
        for diag_col, model in self.models['diag_models'].items():
            model.eval()
            logits = model(X)
            preds = torch.sigmoid(logits)
            diag_probs.append(preds)
        if probs:
            diag_probs = torch.cat(diag_probs, dim=1)
            return diag_probs
        else:
            diag = torch.cat([torch.sigmoid(logits) > 0.5 for logits in diag_probs], dim=1)
            return diag

    def gumbel_sample_contra_obs(self, obs_k, diag_k, treatment_k_contra, obs_k_next, nsamp=1000):
        logits_control = {obs_col: self.get_column_from_tensor(torch.log(obs_k_next), 'obs', obs_col) for obs_col in observation_cols}
        logits_treat = self.predict_obs(obs_k, diag_k, treatment_k_contra, return_logits=True)

        # montecarlo sample gumbel noise
        gx = {}
        for obs_col, logits in logits_control.items():
            mask = ~(logits==-torch.inf).all(axis=1) 
            available_logits = logits[mask]
            g = topdown(available_logits, torch.argmax(available_logits, dim=1), nsamp=nsamp)  # 1000 samples

            g_full = torch.zeros((nsamp, logits.shape[0], logits.shape[1]), device=device)
            g_full[:, mask, :] = g

            gx[obs_col] = g_full

        posteriors = {}
        for obs_col, logits in logits_treat.items():
            posterior_sum = logits + gx[obs_col] # shape: (nsamp, n_obs, n_cat)

            posterior_treat = posterior_sum.argmax(axis=2)

            nsamples, nobservations, nclasses = posterior_sum.shape
            mask = torch.zeros_like(posterior_sum)
            rows = torch.arange(nsamples)[:, None]
            cols = torch.arange(nobservations)[None, :]
            mask[rows, cols, posterior_treat] = 1

            posterior_prob = mask.sum(axis=0) / nsamples # average over samples
            posteriors[obs_col] = posterior_prob

        return torch.cat(list(posteriors.values()), dim=1)

            


        