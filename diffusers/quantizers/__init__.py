class DiffusersQuantizer:
    is_serializable = False


class DiffusersAutoQuantizer:
    @staticmethod
    def merge_quantization_configs(config, override):
        return override if override is not None else config

    @staticmethod
    def from_config(*args, **kwargs):
        raise ValueError("Quantized checkpoints are not supported in the HW5 minimal diffusers subset.")
