import torch

def map_generate(attention_map, pred, p1, p2):
    batches, feaC, feaH, feaW = attention_map.size()

    out_map=torch.zeros_like(attention_map.mean(1))

    for batch_index in range(batches):
        map_tpm = attention_map[batch_index]
        map_tpm = map_tpm.reshape(feaC, feaH*feaW)
        map_tpm = map_tpm.permute([1, 0])
        p1_tmp = p1.permute([1, 0])
        map_tpm = torch.mm(map_tpm, p1_tmp)
        map_tpm = map_tpm.permute([1, 0])
        map_tpm = map_tpm.reshape(map_tpm.size(0), feaH, feaW)

        pred_tmp = pred[batch_index]
        pred_ind = pred_tmp.argmax()
        p2_tmp = p2[pred_ind].unsqueeze(1)

        map_tpm = map_tpm.reshape(map_tpm.size(0), feaH * feaW)
        map_tpm = map_tpm.permute([1, 0])
        map_tpm = torch.mm(map_tpm, p2_tmp)
        out_map[batch_index] = map_tpm.reshape(feaH, feaW)

    return out_map