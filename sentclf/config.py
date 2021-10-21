from functools import partial


def init_func(name, module, *args, **kwargs):
    return partial(getattr(module, name), *args, **kwargs)


def init_obj(name, module, *args, **kwargs):
    return getattr(module, name)(*args, **kwargs)


def update_config(base, config):
    for k, v in config.items():
        if k in base and isinstance(v, dict):
            base[k] = update_config(base[k], v)
        else:
            base[k] = v
    return base
