import torch
import torch.nn as nn


class EdgeAccuracy(nn.Module):
    """
    Measures the accuracy of the edge map
    """

    def __init__(self, threshold=0.5):
        super(EdgeAccuracy, self).__init__()
        self.threshold = threshold

    def __call__(self, inputs, outputs):
        labels = (inputs > self.threshold)
        outputs = (outputs > self.threshold)

        relevant = torch.sum(labels.float())
        selected = torch.sum(outputs.float())

        if relevant == 0 and selected == 0:
            return 1, 1

        true_positive = ((outputs == labels) * labels).float()
        recall = torch.sum(true_positive) / (relevant + 1e-8)
        precision = torch.sum(true_positive) / (selected + 1e-8)

        return precision, recall


class PSNR(nn.Module):
    def __init__(self, max_val):
        super(PSNR, self).__init__()

        base10 = torch.log(torch.tensor(10.0))
        max_val = torch.tensor(max_val).float()

        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20 * torch.log(max_val) / base10)

    def __call__(self, a, b):
        mse = torch.mean((a.float() - b.float()) ** 2)

        if mse == 0:
            return 0

        return self.max_val - 10 * torch.log(mse) / self.base10


class COV(nn.Module):
    def __init__(self, threshold=0.5):
        super(COV, self).__init__()
        self.threshold = threshold

    def __call__(self, inputs, outputs):
        labels = (inputs > self.threshold).float()
        outputs = (outputs > self.threshold).float()

        mean_label = torch.mean(labels)
        mean_outputs = torch.mean(outputs)

        img_a = labels - mean_label
        img_b = outputs - mean_outputs
        cov_value = torch.sum(torch.mul(img_a, img_b)) / (
        torch.sqrt(torch.sum(torch.mul(img_a, img_a)) * torch.sum(torch.mul(img_b, img_b))))

        return cov_value
