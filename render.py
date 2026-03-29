
import os
os.environ['JAX_ENABLE_PLUGIN_DISCOVERY'] = 'false'
os.environ['JAX_PLATFORMS'] = 'cpu'
import jax
import jax.tree_util
import flax
import tyro
import time
import optax
import wandb
import pickle
import random
import wandb_osh
import numpy as np
import flax.linen as nn
import jax.numpy as jnp

from brax import envs
from etils import epath
from dataclasses import dataclass
from typing import NamedTuple, Any
from wandb_osh.hooks import TriggerWandbSyncHook
from flax.training.train_state import TrainState
from flax.linen.initializers import variance_scaling
from brax.io import html

from evaluator import CrlEvaluator
from buffer import TrajectoryUniformSamplingQueue
from train import Actor, SA_encoder, G_encoder, TrainingState, make_env, load_params, Args

@dataclass
class RenderArgs:
    run_dir: str = ""
    vis_length: int = 1000
    num_render: int = 10


def render_policy(training_state, args, save_path):
    """Renders the policy and saves it as an HTML file."""
    eval_env_id = args.eval_env_id if args.eval_env_id else args.env_id
    env = make_env(eval_env_id, args)
    actor = Actor(action_size=env.action_size, network_width=args.actor_network_width, network_depth=args.actor_depth, skip_connections=args.actor_skip_connections, use_relu=args.use_relu)

    @jax.jit
    def policy_step(env_state, actor_params):
        means, _ = actor.apply(actor_params, env_state.obs)
        actions = nn.tanh(means)
        next_state = env.step(env_state, actions)
        return next_state

    rollout_states = []
    for i in range(args.num_render):
        rng = jax.random.key(seed=i+1)
        env_state = env.reset(rng)

        for _ in range(args.vis_length):
            next_state = policy_step(env_state, training_state.actor_state.params)
            rollout_states.append(next_state.pipeline_state)
            env_state = next_state

    # Render and save
    # Convert JAX arrays to numpy for compatibility with older Brax versions
    rollout_states_np = []
    for state in rollout_states:
        converted = jax.tree_util.tree_map(lambda x: np.asarray(x), state)
        rollout_states_np.append(converted)
    
    # Also convert env.sys
    env_sys_np = jax.tree_util.tree_map(lambda x: np.asarray(x), env.sys)
    
    html_string = html.render(env_sys_np, rollout_states_np)
    render_path = f"{save_path}/vis_rerender.html"
    with open(render_path, "w") as f:
        f.write(html_string)
    print(f"Saved rendered policy to {render_path}")


if __name__ == "__main__":
    render_args = tyro.cli(RenderArgs)

    with open(os.path.join(render_args.run_dir, 'args.pkl'), 'rb') as f:
        args = pickle.load(f)

    #make env
    env = make_env(env_id=args.env_id, args=args)
    env = envs.training.wrap(
        env,
        episode_length=args.episode_length,
    )
    obs_size = env.observation_size
    action_size = env.action_size

    key = jax.random.key(args.seed)
    key, actor_key, sa_key, g_key = jax.random.split(key, 4)

    # Network setup
    # Actor
    actor = Actor(action_size=action_size, network_width=args.actor_network_width, network_depth=args.actor_depth, skip_connections=args.actor_skip_connections, use_relu=args.use_relu)
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, np.ones([1, obs_size])),
        tx=optax.adam(learning_rate=args.actor_lr)
    )

    # Critic
    sa_encoder = SA_encoder(network_width=args.critic_network_width, network_depth=args.critic_depth, skip_connections=args.critic_skip_connections, use_relu=args.use_relu)
    sa_encoder_params = sa_encoder.init(sa_key, np.ones([1, args.obs_dim]), np.ones([1, action_size]))
    g_encoder = G_encoder(network_width=args.critic_network_width, network_depth=args.critic_depth, skip_connections=args.critic_skip_connections, use_relu=args.use_relu)
    g_encoder_params = g_encoder.init(g_key, np.ones([1, args.goal_end_idx - args.goal_start_idx]))
    
    critic_state = TrainState.create(
        apply_fn=None,
        params={
            "sa_encoder": sa_encoder_params, 
            "g_encoder": g_encoder_params
            },
        tx=optax.adam(learning_rate=args.critic_lr),
    )

    # Entropy coefficient
    log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
    alpha_state = TrainState.create(
        apply_fn=None,
        params={"log_alpha": log_alpha},
        tx=optax.adam(learning_rate=args.alpha_lr),
    )
    
    # Trainstate
    training_state = TrainingState(
        env_steps=jnp.zeros(()),
        gradient_steps=jnp.zeros(()),
        actor_state=actor_state,
        critic_state=critic_state,
        alpha_state=alpha_state,
    )

    # Load params
    params_path = os.path.join(render_args.run_dir, 'final.pkl')
    alpha_params, actor_params, critic_params = load_params(params_path)

    training_state = training_state.replace(
        actor_state=training_state.actor_state.replace(params=actor_params),
        critic_state=training_state.critic_state.replace(params=critic_params),
        alpha_state=training_state.alpha_state.replace(params=alpha_params)
    )
    
    print("Rendering final policy...", flush=True)
    try:
        render_policy(training_state, args, render_args.run_dir)
    except Exception as e:
        print(f"Error rendering final policy: {e}", flush=True)

