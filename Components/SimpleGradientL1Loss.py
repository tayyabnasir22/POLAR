import torch
import torch.nn as nn
import torch.nn.functional as F

def sobel_filters(img):
    """Compute Sobel gradients for an image tensor (N,C,H,W)."""
    C = img.shape[1]
    sobel_x = torch.tensor([[-1,0,1],
                            [-2,0,2],
                            [-1,0,1]], dtype=img.dtype, device=img.device).view(1,1,3,3)
    sobel_y = torch.tensor([[-1,-2,-1],
                            [ 0, 0, 0],
                            [ 1, 2, 1]], dtype=img.dtype, device=img.device).view(1,1,3,3)

    # Repeat for each channel
    sobel_x = sobel_x.repeat(C, 1, 1, 1)  # [C,1,3,3]
    sobel_y = sobel_y.repeat(C, 1, 1, 1)  # [C,1,3,3]

    gx = F.conv2d(img, sobel_x, padding=1, groups=C)
    gy = F.conv2d(img, sobel_y, padding=1, groups=C)
    return gx, gy

class SimpleGradientL1Loss(nn.Module):
    def __init__(
        self, 
        lambda_l1: float = 1.0,
        lambda_grad: float = 0.05,
        lambda_restore: float = 0.3,
        lambda_boundary: float = 1.0, #TODO tune this if no good training
    ):
        """
        Args:
            lambda_l1:       weight for pixel-wise L1 loss (HR prediction)
            lambda_grad:     weight for gradient loss (HR prediction)
            lambda_restore:  weight for LR depth restoration L1
            lambda_boundary: weight for student->teacher boundary
                             consistency (online distillation). 0 disables
                             the term entirely; it is also skipped
                             automatically whenever either boundary map is
                             absent (eval mode, teacher-only mode, or a
                             model built with build_teacher=False), so the
                             same loss object works for every setup.
        """
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_grad = lambda_grad
        self.lambda_restore = lambda_restore
        self.lambda_boundary = lambda_boundary
        self.l1 = nn.L1Loss()

    def _boundary_consistency(self, student_b, teacher_b):
        # Teacher is frozen; detach defensively so no gradient can ever
        # flow into DINO through this term.
        teacher_b = teacher_b.detach()

        if student_b.shape[-2:] != teacher_b.shape[-2:]:
            student_b = F.interpolate(
                student_b, size=teacher_b.shape[-2:],
                mode='bilinear', align_corners=False,
            )
        return self.l1(student_b, teacher_b)

    def forward(
        self,
        pred, target,
        clean_lr_pred=None,
        clean_lr_gt=None,
        aux=None, # Dict with teach and student boundary
    ):
        # --- Pixel L1 loss ---
        l1_loss = self.l1(pred, target)

        # --- Gradient L1 loss (first-order derivatives) ---
        gx_pred, gy_pred = sobel_filters(pred)
        gx_tgt, gy_tgt = sobel_filters(target)

        grad_loss = self.l1(gx_pred, gx_tgt) + self.l1(gy_pred, gy_tgt)

        total = self.lambda_l1 * l1_loss + self.lambda_grad * grad_loss
        
        # --- LR restoration L1 ---
        if clean_lr_pred is not None and clean_lr_gt is not None:
            restore_loss = self.l1(clean_lr_pred, clean_lr_gt)
            total = total + self.lambda_restore * restore_loss

        # --- Boundary consistency (student vs. frozen DINO teacher) ---
        if (
            self.lambda_boundary > 0
            and aux is not None
            and aux.get('student_boundary') is not None
            and aux.get('teacher_boundary') is not None
        ):
            boundary_loss = self._boundary_consistency(
                aux['student_boundary'], aux['teacher_boundary']
            )
            total = total + self.lambda_boundary * boundary_loss

        return total