from torch import nn


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target = targets
        mel_target.requires_grad = False

        mel_out, mel_out_postnet = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)

        return mel_loss