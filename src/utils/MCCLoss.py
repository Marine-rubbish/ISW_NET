import torch
import torch.nn as nn

class MCCLoss(nn.Module):
    """
    MCCLoss类实现了基于 Matthews 相关系数的损失函数。

    参数
    ---
    无
    
    功能
    ---
    计算 Matthews 相关系数的损失。

    示例
    ---
    ```python
    criterion = MCCLoss()    # 创建 MCCLoss 实例
    loss = criterion(logits, labels)    # 计算损失
    ```
    """
    def __init__(self):
        super(MCCLoss, self).__init__()

    def forward(self, logits, labels):
        logits = torch.sigmoid(logits)
        labels = labels.float()
        
        tp = (logits * labels).sum(dim=(1, 2, 3))
        tn = ((1 - logits) * (1 - labels)).sum(dim=(1, 2, 3))
        fp = (logits * (1 - labels)).sum(dim=(1, 2, 3))
        fn = ((1 - logits) * labels).sum(dim=(1, 2, 3))
        
        numerator = (tp * tn - fp * fn)
        denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = numerator / (denominator + 1e-5)
        
        return 1 - mcc.mean()