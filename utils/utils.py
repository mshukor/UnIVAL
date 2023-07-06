


def print_trainable_params_percentage(model):


    orig_param_size = sum(p.numel() for p in model.parameters())

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    trainable_size = count_parameters(model)

    percentage = trainable_size / orig_param_size * 100

    print(f"Trainable param percentage: {percentage:.2f}% ({trainable_size}/{orig_param_size})")

    return percentage



def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print