from collections import defaultdict
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

# HyperPPO-specific import: Graph HyperNetwork-based actor
from model.core import hyperActor  # <-- reference to your HyperPPO code

@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, torch.backends.cudnn.deterministic=True"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out videos folder)"""
    save_model: bool = True
    """whether to save model into the runs/{run_name} folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""

    # Algorithm specific arguments
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""

    # Turned this down from 1e-4 to 1e-5
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 512
    """the number of parallel environments"""
    num_eval_envs: int = 8
    """the number of parallel evaluation environments"""
    partial_reset: bool = True
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 50
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 50
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use for the environment"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.8
    """the discount factor gamma"""
    gae_lambda: float = 0.9
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""

    target_kl: float = 0.1
    """the target KL divergence threshold"""
    reward_scale: float = 1.0
    """Scale the reward by this factor"""
    eval_freq: int = 25
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    finite_horizon_gae: bool = False


    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb

    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            import wandb
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)

    def close(self):
        self.writer.close()

###############################################################################
#                            HyperPPO Agent                                  #
###############################################################################
class HyperAgent(nn.Module):
    """
    Replaces the fixed-architecture MLP with a HyperPPO-based GHN actor
    plus a standard MLP critic for value estimation.
    """
    def __init__(self, envs):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obs_dim = np.prod(envs.single_observation_space.shape)
        act_dim = np.prod(envs.single_action_space.shape)

        # -- Critic (Value Function) as a simple MLP --
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        # -- HyperNetwork-based Actor (Policy) --
        # allowable_layers = [16, 32, 64, 128, 256]
        allowable_layers = [16, 32, 64]
        self.hyper_actor = hyperActor(
            act_dim=act_dim,
            obs_dim=obs_dim,
            allowable_layers=allowable_layers,
            meta_batch_size=8,            # Sample multiple architectures at once
            device=self.device,
            architecture_sampling_mode="biased",
            multi_gpu=False,              # set True if using multiple GPUs
            std_mode='single',  # can be 'single', 'multi', or 'arch_conditioned'
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Return the value function estimate from the MLP critic."""
        return self.critic(x)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action_in: torch.Tensor = None
    ):
        """
        Generate action & log-probs from GHN-based actor,
        also return entropy & MLP critic value for PPO updates.
        """
        # We call "change_graph()" to sample a new architecture (or re-use one).
        # self.hyper_actor.change_graph(repeat_sample=False)

        # Forward pass through GHN-based actor
        mu, log_std = self.hyper_actor(obs, track=False)
        std = log_std.exp()
        dist = Normal(mu, std)

        if action_in is None:
            action_out = dist.sample()
        else:
            action_out = action_in

        log_prob = dist.log_prob(action_out).sum(axis=1)
        entropy = dist.entropy().sum(axis=1)
        value_estimate = self.get_value(obs)

        return action_out, log_prob, entropy, value_estimate

###############################################################################
#                             Main Training Script                            #
###############################################################################
if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    if args.exp_name is None:
        # e.g. "PickCube-v1__ppo__<seed>__<timestamp>"
        args.exp_name = os.path.basename(__file__)[:-3]
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Environment setup
    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="physx_cuda")
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode

    envs = gym.make(
        args.env_id,
        num_envs=args.num_envs if not args.evaluate else 1,
        reconfiguration_freq=args.reconfiguration_freq,
        **env_kwargs
    )
    eval_envs = gym.make(
        args.env_id,
        num_envs=args.num_eval_envs,
        reconfiguration_freq=args.eval_reconfiguration_freq,
        **env_kwargs
    )

    # Flatten Dict action spaces if needed
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)

    # Video capture
    if args.capture_video:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")

        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x: (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(
                envs,
                output_dir=f"runs/{run_name}/train_videos",
                save_trajectory=False,
                save_video_trigger=save_video_trigger,
                max_steps_per_video=args.num_steps,
                video_fps=30
            )
        eval_envs = RecordEpisode(
            eval_envs,
            output_dir=eval_output_dir,
            save_trajectory=args.evaluate,
            trajectory_name="trajectory",
            max_steps_per_video=args.num_eval_steps,
            video_fps=30
        )

    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)

    # Setup logging
    logger = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb
            config = vars(args)
            config["env_cfg"] = dict(
                **env_kwargs,
                num_envs=args.num_envs,
                env_id=args.env_id,
                reward_mode="dense",
                env_horizon=max_episode_steps,
                partial_reset=args.partial_reset
            )
            config["eval_env_cfg"] = dict(
                **env_kwargs,
                num_envs=args.num_eval_envs,
                env_id=args.env_id,
                reward_mode="dense",
                env_horizon=max_episode_steps,
                partial_reset=False
            )
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                save_code=True,
                group="PPO",
                tags=["ppo", "HyperPPO"]
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % (
                "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])
            ),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    ########################################################################
    # Replace the old Agent with the HyperAgent
    ########################################################################
    agent = HyperAgent(envs).to(device)

    # If resuming from checkpoint
    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint, map_location=device))

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # PPO storage
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)

    print("####")
    print(f"args.num_iterations={args.num_iterations} "
          f"args.num_envs={args.num_envs} "
          f"args.num_eval_envs={args.num_eval_envs}")
    print(f"args.minibatch_size={args.minibatch_size} "
          f"args.batch_size={args.batch_size} "
          f"args.update_epochs={args.update_epochs}")
    print("####")

    action_space_low = torch.from_numpy(envs.single_action_space.low).to(device)
    action_space_high = torch.from_numpy(envs.single_action_space.high).to(device)

    def clip_action(action: torch.Tensor):
        return torch.clamp(action.detach(), action_space_low, action_space_high)


    # def clip_action(action: torch.Tensor):
    #     print(f"Raw Action (Before Clipping): {action.mean().item()}, Min: {action.min().item()}, Max: {action.max().item()}")

    #     # Keep the gripper fixed at 4cm width (Panda Robot)
    #     action[:, -2:] = torch.clamp(action[:, -2:], 0.02, 0.04)  # Gripper range [0.02, 0.04]

    #     action = torch.clamp(action, action_space_low + 0.01, action_space_high - 0.01)

    #     print(f"Clipped Action (After Clipping): {action.mean().item()}, Min: {action.min().item()}, Max: {action.max().item()}")
    #     return action


    # def clip_action(action: torch.Tensor):
    #     action[:, -2:] = 0.04 
    #     return torch.clamp(action.detach(), action_space_low+0.01, action_space_high-0.01)

    for iteration in range(1, args.num_iterations + 1):
        print(f"Epoch: {iteration}, global_step={global_step}")
        print(f"Sampling new architecture")
        agent.hyper_actor.change_graph(repeat_sample=False)

        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)

        # Evaluate periodically
        agent.eval()
        if iteration % args.eval_freq == 1:
            print("Evaluating")
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    # Use deterministic actions for eval
                    det_action, _, _, _ = agent.get_action_and_value(eval_obs)
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(
                        clip_action(det_action)
                    )
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
            print(f"Evaluated {args.num_eval_steps * args.num_eval_envs} steps resulting in {num_episodes} episodes")
            for k, v in eval_metrics.items():
                mean_ = torch.stack(v).float().mean()
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean_, global_step)
                print(f"eval_{k}_mean={mean_}")
            if args.evaluate:
                break

        # Save intermediate checkpoint
        if args.save_model and iteration % args.eval_freq == 1:
            model_path = f"runs/{run_name}/ckpt_{iteration}.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")

        # Possibly anneal learning rate
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Collect rollouts
        rollout_time = time.time()
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            agent.train()  # actor in training mode for sampling
            with torch.no_grad():
                act, logp, ent, val = agent.get_action_and_value(next_obs)
                actions[step] = act
                logprobs[step] = logp
                values[step] = val.flatten()

            # Step environment
            next_obs, reward, terminations, truncations, infos = envs.step(clip_action(actions[step]))
            next_done = torch.logical_or(terminations, truncations).float().to(device)
            rewards[step] = reward.view(-1) * args.reward_scale
            # print(f"Reward: {reward.mean().item()}, Min: {reward.min().item()}, Max: {reward.max().item()}")

            # Log episodic returns
            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k, v in final_info["episode"].items():
                    if logger is not None:
                        logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)
                # For partial resets: store final state values
                with torch.no_grad():
                    final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = agent.get_value(
                        infos["final_observation"][done_mask]
                    ).view(-1)

        rollout_time = time.time() - rollout_time

        # GAE or finite-horizon advantage
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t]

                if args.finite_horizon_gae:
                    # optional specialized GAE code
                    pass
                else:
                    delta = rewards[t] + args.gamma * real_next_values - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam

            returns = advantages + values

        # Flatten batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # PPO Update
        agent.train()
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        update_time = time.time()

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Fresh tensors for each minibatch
                obs_batch = b_obs[mb_inds]
                act_batch = b_actions[mb_inds]
                logprob_batch = b_logprobs[mb_inds]
                adv_batch = b_advantages[mb_inds]
                ret_batch = b_returns[mb_inds]
                val_batch = b_values[mb_inds]

                # Forward pass
                _, new_logprob, entropy, new_value = agent.get_action_and_value(obs_batch, act_batch)
                logratio = new_logprob - logprob_batch
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()

                # Normalize advantage
                if args.norm_adv:
                    adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)

                # Policy loss
                pg_loss1 = -adv_batch * ratio
                pg_loss2 = -adv_batch * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                new_value = new_value.view(-1)
                if args.clip_vloss:
                    v_unclipped = (new_value - ret_batch) ** 2
                    v_clipped = val_batch + torch.clamp(new_value - val_batch, -args.clip_coef, args.clip_coef)
                    v_clipped = (v_clipped - ret_batch) ** 2
                    v_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_value - ret_batch) ** 2).mean()

                # Entropy bonus
                entropy_loss = entropy.mean()

                # Total loss
                total_loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        update_time = time.time() - update_time

        # Logging
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var((y_true - y_pred)) / var_y

        if logger is not None:
            logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
            logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            logger.add_scalar("losses/approx_kl", approx_kl, global_step)
            logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            logger.add_scalar("losses/explained_variance", explained_var, global_step)
            sps = int(global_step / (time.time() - start_time))
            print(f"SPS: {sps}, pg_loss={pg_loss.item():.4f}, v_loss={v_loss.item():.4f}")
            logger.add_scalar("charts/SPS", sps, global_step)
            logger.add_scalar("time/step", global_step, global_step)
            logger.add_scalar("time/update_time", update_time, global_step)
            logger.add_scalar("time/rollout_time", rollout_time, global_step)
            logger.add_scalar("time/rollout_fps", args.num_envs * args.num_steps / rollout_time, global_step)

    # Final Save
    if not args.evaluate:
        if args.save_model and logger is not None:
            model_path = f"runs/{run_name}/final_ckpt.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        if logger is not None:
            logger.close()

    envs.close()
    eval_envs.close()