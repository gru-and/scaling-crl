# .jax_config.py
import jax
jax.config.update('jax_disable_most_optimizations', True)
jax.config.update('jax_enable_x64', False)
jax.config.update('jax_disable_jit', False)
jax.config.update('jax_enable_plugin_discovery', False)
jax.config.update('jax_platforms', 'cuda')
jax.config.update('jax_disable_most_optimizations', True)