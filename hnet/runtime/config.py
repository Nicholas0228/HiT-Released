"""Use the checkpoint's declared execution profile in training and sampling."""
_UNSPECIFIED = object()


def execution_options(training):
    backend = training.get('execution_backend', 'native')
    budget = training.get('activation_memory_budget')
    compile_enabled = bool(training.get('compile', True))
    if backend not in ('native', 'hopper'):
        raise ValueError(f'Unknown execution profile: {backend}')
    if backend == 'hopper' and training.get('routing_diagnostics_at_log_steps', False):
        raise ValueError('Graph-matched Hopper execution requires routing diagnostics on every update')
    if backend == 'hopper' and not compile_enabled:
        raise ValueError('The Hopper execution profile requires compile=true')
    if backend == 'native' and budget is not None:
        raise ValueError('An activation budget requires the explicit Hopper execution profile')
    return backend, budget, compile_enabled


def execution_identity(config, training):
    backend, budget, _ = execution_options(training)
    if backend == 'native':
        return None
    from .hopper import runtime_identity
    return runtime_identity(config, budget)


def configure_model_execution(model, training, *, expected=_UNSPECIFIED):
    """Configure before first use; sampling must supply saved identity, even None.

    Native checkpoints keep their original optional compile behavior. A Hopper
    checkpoint cannot silently fall back to native or load another FA3 build.
    """
    backend, budget, compile_enabled = execution_options(training)
    identity = execution_identity(model.config, training)
    if expected is not _UNSPECIFIED and identity != expected:
        raise ValueError('Execution profile does not match the checkpoint identity')
    if compile_enabled:
        model.compile_blocks(runtime=backend, activation_memory_budget=budget)
    return identity
