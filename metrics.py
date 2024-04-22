from torchmetrics import JaccardIndex
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Iou_score(outputs,masks):
    iou_metric = JaccardIndex(task='binary').to(device)
    pred_masks = torch.argmax(outputs, dim=1)
    iou_metric(outputs, masks)
    return iou_metric.compute()