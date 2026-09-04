"""
Knowledge Distillation Engine for SYMBEX-1 models.
Distills knowledge from an FP32 Teacher into a 1-bit Student using KL-Divergence.
"""
import torch
import torch.nn as nn
from .models.core import BipolarStepSTE

def train_teacher(teacher, X_train, y_train, epochs=150, lr=0.005):
    """
    Trains the master FP32 model.

    Args:
        teacher (nn.Module): FP32 model to train.
        X_train (torch.Tensor): Training data.
        y_train (torch.Tensor): Training labels.
        epochs (int): Number of epochs.
        lr (float): Learning rate.

    Returns:
        nn.Module: Trained teacher model.
    """
    opt = torch.optim.Adam(teacher.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    
    teacher.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = crit(teacher(X_train), y_train)
        loss.backward()
        opt.step()
        
    return teacher

def distill_student(student, teacher, X_train, y_train, epochs=300, lr=0.001, T=4.0, alpha=0.85):
    """
    Distills knowledge from the FP32 Teacher into the quantized Student.
    Uses a hybrid loss: CrossEntropy (ground truth) + KL-Divergence (teacher softness).

    Args:
        student (nn.Module): Quantized (1-bit) model to train.
        teacher (nn.Module): Trained FP32 master model.
        X_train (torch.Tensor): Training data.
        y_train (torch.Tensor): Training labels.
        epochs (int): Number of epochs (Should be 2x the Teacher's epochs).
        lr (float): Learning rate.
        T (float): Softmax Temperature for KL-Divergence.
        alpha (float): Weight ratio for KL-Divergence vs CrossEntropy.

    Returns:
        nn.Module: Trained student model.
    """
    s_opt = torch.optim.Adam(student.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    kl_div = nn.KLDivLoss(reduction='batchmean')
    
    teacher.eval()
    
    for _ in range(epochs):
        student.train()
        s_opt.zero_grad()
        
        with torch.no_grad():
            t_out = teacher(X_train)
            
        s_out = student(X_train)
        
        # Calculate Teacher-Student divergence (Knowledge Transfer)
        loss_kl = kl_div(
            nn.functional.log_softmax(s_out / T, dim=1),
            nn.functional.softmax(t_out / T, dim=1)
        ) * (T * T)
        
        # Calculate standard target loss (Ground Truth)
        loss_ce = crit(s_out, y_train)
        
        # Hybrid Loss
        loss = alpha * loss_kl + (1.0 - alpha) * loss_ce
        
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        s_opt.step()
        
    return student

def evaluate_accuracy(model, X_test, y_test):
    """
    Standard PyTorch evaluation metric.
    """
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(X_test), dim=1)
        accuracy = (preds == y_test).float().mean().item() * 100.0
    return accuracy
