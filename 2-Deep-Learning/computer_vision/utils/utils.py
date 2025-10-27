import torch
from sklearn.metrics import roc_auc_score
import json
import pickle
import os
from tqdm import tqdm

def save_checkpoint(state, filename= 'my_checkpoint.pth.tar'):
    print('=> saving checkpoint')
    torch.save(state, filename)

def load_checkpoint(model, optimizer, checkpoint):
    print('=> Loading Checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer

def evaluate_classification_model(y_true, predictions, labels):
    auc_scores = roc_auc_score(y_true, predictions, average=None)
    auc_score_macro = roc_auc_score(y_true, predictions, average='macro')
    auc_score_micro = roc_auc_score(y_true, predictions, average='micro')
    auc_score_weighted = roc_auc_score(y_true, predictions, average='weighted')
    results = {
    "groun_truth" : y_true,
    "predictions" : predictions,
    "labels" : labels,
    "auc_scores" : auc_scores,
    "auc_macro" : auc_score_macro,
    "auc_micro" : auc_score_micro,
    "auc_weighted" : auc_score_weighted,
    }
    return results

def add_data(path, uuid, result):
    if not os.path.exists(path):
        with open(path, 'w') as f:
            json.dump({}, f)
    
    with open(path, 'r+') as f:
        dados = json.load(f)
        dados[uuid] = result
        f.seek(0)
        json.dump(dados, f)
        f.truncate()

def store_test_metrics(var, path, filename="test_metrics", name="", json=False, result_path='result.json'):
    with open(f"{path}/{filename}", "wb") as f:
        pickle.dump(var, f) #salva arquivo
    if(json==True):
        auc_macro = var['auc_macro']
        add_data(path=result_path, uuid=name, result=auc_macro)
        pass

def check_auc(loader, model, device, labels, save= False, Train= True):
    if Train:
        print('Checking auc on training data')
    else:
        print('Checking auc on val/test data')
    model.eval()
    trues, preds_all = [], []
    loop = tqdm(enumerate(loader), total=len(loader), leave=False)
    for i, (data, targets) in loop:
        data = data.to(device)
        targets.to(device)
        scores = model(data)
        preds = torch.sigmoid(scores)
        trues.extend(targets.cpu().tolist())
        preds_all.extend(preds.cpu().tolist())
    results = evaluate_classification_model(trues, preds_all, labels)
    auc = results['auc_scores']
    auc_macro = results['auc_macro']
    auc_micro = results['auc_micro']
    auc_weighted = results['auc_weighted']
    print(f'AUC: {auc} Auc-Macro: {auc_macro} Auc-Micro: {auc_micro} Auc-Weighted: {auc_weighted}')
    if save:
        store_test_metrics(results, path='/home/matheus_levy/workspace/pytorch_study/metrics',
                            filename=f"metrics_global", name='model', json=True, result_path='/home/matheus_levy/workspace/pytorch_study/metrics/result.json')
