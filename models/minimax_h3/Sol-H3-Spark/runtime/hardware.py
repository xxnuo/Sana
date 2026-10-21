"""Device selection for the single-GPU Spark and Thor runtimes."""


def require_supported_gpu(torch):
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Sol-H3 requires one visible GPU")
    capability = tuple(torch.cuda.get_device_capability())
    if capability not in {(11, 0), (12, 1)}:
        raise RuntimeError("Sol-H3 requires one Thor SM110 or Spark SM121 GPU")
    return capability


def bsa_kernel_module(capability):
    if capability in {(11, 0), (12, 1)}:
        return "csrc.fwd.sm120_blk64.bsa_fwd_sm120"
    raise ValueError(f"Unsupported BSA device: {capability}")
