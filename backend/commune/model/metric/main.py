import torch
import torch.nn as nn

from .dilate import dilate_loss
EPS = 1E-10

def diff_root_loss(input, target, loss_fn=nn.SmoothL1Loss(reduce=False), reduce_dims =[1]):
  """
  Calculates the loss for when the target and pred are predicting the wrong direction

  input: input tensor (N,*)
  target: target tensor (N,*)
  loss_fn: non reducing function
  """

  # create an element mask across the input/target
  element_mask = (input*target > 0).float()
  element_loss = loss_fn(input=input, target=target)

  masked_element_loss = element_loss*element_mask
  final_loss = (masked_element_loss.sum(reduce_dims))/(element_mask.sum(reduce_dims)+EPS)

  return final_loss

def time_series_loss(input, target, bounds = [-7,None]):
  _, seq_len =  target.shape
  min_b, max_b = bounds
  if max_b is None:
    input = input[:, min_b:]
    target = target[:, min_b:]
  else:
    assert(max_b > min_b)
    input = input[:, min_b: max_b]
    target = target[:, min_b: max_b]

  return torch.nn.SmoothL1Loss()(input=input, target=target)


def compounding_future_penalty(input, target, factor=1.1):
  loss = nn.SmoothL1Loss(reduction="none")(input,target)

  future_magnifier = torch.cumprod(torch.full_like(input, fill_value=factor), dim=1)

  return (loss*future_magnifier).mean()

def auto_corr_mse(y_pred, y_target, shift=1):

  y_pred_diff = y_pred[:,shift:]-y_pred[:,:-shift]
  y_target_diff = y_target[:, shift:] - y_target[:, :-shift]


  return ((y_pred_diff-y_target_diff)**2).mean()


def past_loss_fn(input, target, fn = nn.SmoothL1Loss()):

  return fn(input=input, target=target[:, -input.shape[1]:])

def time_cl_loss(input, target, fn=nn.SmoothL1Loss(reduce=False), beta=0.1, shift_factor=0.1, time_step=10):

  shift = shift_factor * time_step
  seq_len = input.shape[1]

  time_weight =torch.exp(-beta * (torch.arange(seq_len).float() - shift) ** 2).unsqueeze(0).cuda()
  loss_per_element = fn(input=input, target=target)
  weighted_loss_per_element = loss_per_element*time_weight

  out_loss =  weighted_loss_per_element.mean()

  return out_loss

def shift_mse(y_pred, y_current_target, y_future_target, shift=2, offset=2):

  y_pred = y_pred[:,offset:]
  y_target = torch.cat([y_current_target, y_future_target], dim=1)
  y_target = y_target[:, offset+shift:(offset+shift)+y_pred.shape[1]]

  return ((y_pred.squeeze(-1)-y_target)**2).mean()


def mean_negative_log_likelihood(input_mean, input_cov, target):

  mnll = 0.5*torch.log(input_cov**2) + 0.5*((input_mean - target)**2)/(input_cov**2)
  return mnll.mean()


def quantile_loss(input, target,q):
  error = target - input
  return torch.max(q*error, (q-1)*error).mean()


def compute_smape(y_pred, y_target, reduce_dims = [], reduce_fn = torch.mean):
  with torch.no_grad():
    out_val =  (2 * (y_target - y_pred).abs() / (y_target.abs() + y_pred.abs()))
    if reduce_dims:
      out_val = reduce_fn(out_val, dim=reduce_dims)
  return out_val.cpu()

def compute_map(y_pred, y_target, reduce_dims = [], reduce_fn = torch.mean):
  with torch.no_grad():
    out_val =  ((y_pred - y_target).abs() / y_target.abs())
    if reduce_dims:
      out_val = reduce_fn(out_val, dim=reduce_dims)
  return out_val.cpu()

def compute_mse(y_pred, y_target, reduce_dims = [], reduce_fn = torch.mean):
  with torch.no_grad():
    out_val = ((y_pred - y_target) ** 2) ** 0.5
    if reduce_dims:
      out_val = reduce_fn(out_val, dim=reduce_dims)
  return out_val.cpu()

def compute_mae(y_pred, y_target, reduce_dims = [], reduce_fn = torch.mean):
  with torch.no_grad():
    out_val = (y_pred - y_target).abs()
    if reduce_dims:
      out_val = reduce_fn(out_val, dim=reduce_dims)
  return out_val.cpu()
import torch
import torch.nn as nn



def time_series_loss(input, target, bounds = [-7,None]):
  _, seq_len =  target.shape
  min_b, max_b = bounds
  if max_b is None:
    input = input[:, min_b:]
    target = target[:, min_b:]
  else:
    assert(max_b > min_b)
    input = input[:, min_b: max_b]
    target = target[:, min_b: max_b]

  return torch.nn.SmoothL1Loss()(input=input, target=target)


def compounding_future_penalty(input, target, factor=1.1):
  loss = nn.SmoothL1Loss(reduction="none")(input,target)

  future_magnifier = torch.cumprod(torch.full_like(input, fill_value=factor), dim=1)

  return (loss*future_magnifier).mean()

def auto_corr_mse(y_pred, y_target, shift=1):

  y_pred_diff = y_pred[:,shift:]-y_pred[:,:-shift]
  y_target_diff = y_target[:, shift:] - y_target[:, :-shift]


  return ((y_pred_diff-y_target_diff)**2).mean()


def past_loss_fn(input, target, fn = nn.SmoothL1Loss()):

  return fn(input=input, target=target[:, -input.shape[1]:])

def time_cl_loss(input, target, fn=nn.SmoothL1Loss(reduce=False), beta=0.1, shift_factor=0.1, time_step=10):

  shift = shift_factor * time_step
  seq_len = input.shape[1]

  time_weight =torch.exp(-beta * (torch.arange(seq_len).float() - shift) ** 2).unsqueeze(0).cuda()
  loss_per_element = fn(input=input, target=target)
  weighted_loss_per_element = loss_per_element*time_weight

  out_loss =  weighted_loss_per_element.mean()

  return out_loss

def shift_mse(y_pred, y_current_target, y_future_target, shift=2, offset=2):

  y_pred = y_pred[:,offset:]
  y_target = torch.cat([y_current_target, y_future_target], dim=1)
  y_target = y_target[:, offset+shift:(offset+shift)+y_pred.shape[1]]

  return ((y_pred.squeeze(-1)-y_target)**2).mean()


def mean_negative_log_likelihood(input_mean, input_cov, target):

  mnll = 0.5*torch.log(input_cov**2) + 0.5*((input_mean - target)**2)/(input_cov**2)
  return mnll.mean()


def quantile_loss(input, target,q):
  error = target - input
  return torch.max(q*error, (q-1)*error).mean()


def compute_smape(y_pred, y_target, reduce_dims = [], reduce_fn = torch.mean):
  with torch.no_grad():
    out_val =  (2 * (y_target - y_pred).abs() / (y_target.abs() + y_pred.abs()))
    if reduce_dims:
      out_val = reduce_fn(out_val, dim=reduce_dims)
  return out_val.cpu()

def compute_map(y_pred, y_target, reduce_dims = [], reduce_fn = torch.mean):
  with torch.no_grad():
    out_val =  ((y_pred - y_target).abs() / y_target.abs())
    if reduce_dims:
      out_val = reduce_fn(out_val, dim=reduce_dims)
  return out_val.cpu()

def compute_mse(y_pred, y_target, reduce_dims = [], reduce_fn = torch.mean):
  with torch.no_grad():
    out_val = ((y_pred - y_target) ** 2) ** 0.5
    if reduce_dims:
      out_val = reduce_fn(out_val, dim=reduce_dims)
  return out_val.cpu()

def compute_mae(y_pred, y_target, reduce_dims = [], reduce_fn = torch.mean):
  with torch.no_grad():
    out_val = (y_pred - y_target).abs()
    if reduce_dims:
      out_val = reduce_fn(out_val, dim=reduce_dims)
  return out_val.cpu()

def single_quantile_loss( y_pred: torch.Tensor,
                 target: torch.Tensor,
                 quantile: float) -> torch.Tensor:
    """
    Quantile loss, i.e. a quantile of ``q=0.5`` will give half of the mean absolute error as it is calcualted as

    Defined as ``max(q * (y-y_pred), (1-q) * (y_pred-y))``
    """
    # calculate quantile loss
    errors = target - y_pred
    loss = (torch.max((quantile - 1) * errors, quantile * errors).unsqueeze(-1)).mean()
    return loss


def smooth_l1_loss(input, target,reduce_dims=[0,1], reduce_mode='mean'):
  loss = nn.SmoothL1Loss(reduce=False)

  assert reduce_mode in ['mean', 'sum']

  per_element_loss =  loss(input=input, target=target)

  reduced_loss = getattr(torch, reduce_mode)(per_element_loss, dim=reduce_dims)

  return reduced_loss