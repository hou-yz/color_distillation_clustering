import torch.nn as nn
import torch.nn.functional as F


class KD_loss(nn.Module):
    def __init__(self, temperature=1):
        super(KD_loss, self).__init__()
        self.temperature = temperature

    def forward(self, student_output, teacher_output):
        p = F.log_softmax(student_output / self.temperature, dim=1)
        q = F.softmax(teacher_output / self.temperature, dim=1)
        l_kl = F.kl_div(p, q)  # forward KL
        return l_kl
