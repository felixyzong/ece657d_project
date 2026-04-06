import torch.nn.functional as F


def kd_loss(student_logits, teacher_logits, labels, alpha=0.5, temperature=4.0):
    ce_loss = F.cross_entropy(student_logits, labels)

    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

    return (1 - alpha) * ce_loss + alpha * (temperature * temperature) * distill_loss
