"""Record actual GPU variants and optionally require a frozen hardware model."""
def validate_hardware(cfg):
    import torch
    required=cfg.get('required_gpu_name')
    if required is not None and (not isinstance(required,str) or not required.strip()):raise ValueError('required_gpu_name must be a nonempty exact name or null')
    cuda=str(cfg.get('device','cpu')).startswith('cuda')
    name=torch.cuda.get_device_name(cfg['device']) if cuda and torch.cuda.is_available() else None
    if required is not None and name!=required:raise RuntimeError('GPU hardware contract failed: expected '+required+', got '+str(name))
    return dict(gpu=name,required_gpu_name=required,torch=str(torch.__version__),cuda_runtime=torch.version.cuda)
