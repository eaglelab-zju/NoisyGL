import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def estimate_C(model, x, adj, n_classes, est_mode="max"):
    '''
        Estimate C from pretrained model.
        model: model pretrained on noisy data.
        graphs: training input graphs.
        anchors: list or dict of nodes with exact label.
        C: exact corruption matrix.
        est_mode: mode of estimation
    '''
    scores, idx = model(x, adj).max(dim=1)
    candidates = dict()
    min_val = torch.min(scores)
    max_val = torch.max(scores)
    temp = torch.ones_like(scores)
    for class_id in range(n_classes):
        if est_mode == "max":
            cand_id = torch.argmax(torch.where(idx==class_id,\
                                               scores,\
                                               temp*min_val))
        elif est_mode == "min":
            cand_id = torch.argmin(torch.where(idx==class_id,\
                                               scores,\
                                               temp*max_val))
        else:
            raise NotImplementedError("Should there be a better mode?")
        candidates[class_id] = cand_id
    return _C(model, x, adj, n_classes, candidates)


def _C(model, x, adj, n_classes, candidates):
    '''
        Internal function to return the corruption matrix.

        model: pretrained model on noisy data
        candidates: dictionary from label to a representative sample
                    (TODO extension: list of samples)
    '''
    softmax = nn.Softmax(dim=1)
    C = np.zeros((n_classes, n_classes), dtype=float)
    model.eval()
    scores = model(x, adj)
    for class_id, cand_id in candidates.items():
        cand_score = scores[cand_id].unsqueeze(dim=0)
        probs = softmax(cand_score)
        C[class_id] = probs.detach().cpu().numpy()
    # print("Estimated C: \n", C)
    return C


def forward_correction_xentropy(output, labels, C, device, nclass):
    '''
        Forward loss correction. In cross-entropy, softmax is the inverse
        link function.

        output: raw (logits) output from model
        labels: true labels
        C: correction matrix
    '''
    C = C.astype(np.float32)
    C = torch.from_numpy(C).to(device)
    softmax = nn.Softmax(dim=1)
    label_oh = torch.FloatTensor(len(labels), nclass).to(device)
    label_oh.zero_()
    label_oh.scatter_(1,labels.view(-1,1),1)
    output = softmax(output)
    output = torch.clamp(output, min=1e-5, max=1.0-1e-5)
    return -torch.mean(label_oh * torch.log(torch.matmul(output, C)))


def backward_correction(output, labels, C):
    '''
        Backward loss correction.

        output: raw (logits) output from model
        labels: true labels
        C: correction matrix
    '''
    softmax = nn.Softmax(dim=1)
    # C can be a singular matrix, so we use the generalized inverse matrix
    C_inv = np.linalg.pinv(C).astype(np.float32)
    # C_inv = np.linalg.inv(C).astype(np.float32)
    C_inv = torch.from_numpy(C_inv).to(output.device)
    label_oh = torch.FloatTensor(len(labels), output.shape[1]).to(output.device)
    label_oh.zero_()
    label_oh.scatter_(1, labels.view(-1,1),1)

    # output = softmax(output)
    # output = torch.clamp(output, min=1e-5, max=1.0 - 1e-5)
    # loss = -torch.mean(torch.matmul(label_oh, C_inv) * torch.log(output))
    # output0 = output
    output = F.log_softmax(output, dim=1)
    loss = -torch.mean(torch.matmul(label_oh, C_inv) * output)
    return loss
