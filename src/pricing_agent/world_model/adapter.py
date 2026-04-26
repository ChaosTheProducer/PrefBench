"""World-model adapters for the pricing negotiation environment.

This module provides:
1) A deterministic codec for flattening `(move_idx, delta_idx)` into one
   discrete action index.
2) A Gymnasium environment wrapper that exposes a flat `Discrete` action
   space on top of `PPOPricingEnv`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from pricing_agent.ppo_env import PPOPricingEnv
from pricing_env.gym_wrapper import GYMNASIUM_AVAILABLE

if GYMNASIUM_AVAILABLE:
    import gymnasium as gym
    from gymnasium import spaces
else:  # pragma: no cover - optional dependency path
    gym = object  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]


@dataclass(frozen=True)
class DreamerActionCodec:
    """Deterministic codec between tuple action and flat discrete action id.

    Attributes:
        move_count: Number of move categories.
        delta_bin_count: Number of discrete delta bins.
    """

    move_count: int
    delta_bin_count: int

    def __post_init__(self) -> None:
        """Validates codec dimensions."""

        if int(self.move_count) <= 0:
            raise ValueError("`move_count` must be positive.")
        if int(self.delta_bin_count) <= 0:
            raise ValueError("`delta_bin_count` must be positive.")

    @property
    def action_size(self) -> int:
        """Returns total flattened action size."""

        return int(self.move_count) * int(self.delta_bin_count)

    def flatten(self, *, move_idx: int, delta_idx: int) -> int:
        """Flattens tuple action into one integer id.

        Args:
            move_idx: Move token index.
            delta_idx: Price-delta bin index.

        Returns:
            Flat action index in `[0, action_size)`.
        """

        self._validate_tuple(move_idx=move_idx, delta_idx=delta_idx)
        return int(move_idx) * int(self.delta_bin_count) + int(delta_idx)

    def unflatten(self, action_idx: int) -> Tuple[int, int]:
        """Decodes flat action id into `(move_idx, delta_idx)`.

        Args:
            action_idx: Flat action index.

        Returns:
            Tuple of `(move_idx, delta_idx)`.
        """

        action_idx = int(action_idx)
        if action_idx < 0 or action_idx >= self.action_size:
            raise ValueError(f"Invalid flattened action index: {action_idx}.")
        move_idx = action_idx // int(self.delta_bin_count)
        delta_idx = action_idx % int(self.delta_bin_count)
        return int(move_idx), int(delta_idx)

    def _validate_tuple(self, *, move_idx: int, delta_idx: int) -> None:
        """Validates tuple action indices.

        Args:
            move_idx: Move token index.
            delta_idx: Price-delta bin index.
        """

        move_idx = int(move_idx)
        delta_idx = int(delta_idx)
        if move_idx < 0 or move_idx >= int(self.move_count):
            raise ValueError(f"Invalid move index: {move_idx}.")
        if delta_idx < 0 or delta_idx >= int(self.delta_bin_count):
            raise ValueError(f"Invalid delta index: {delta_idx}.")


class DreamerDiscretePricingEnv(gym.Env if GYMNASIUM_AVAILABLE else object):
    """Discrete-action view of `PPOPricingEnv` for world-model agents.

    Action interface is changed from `MultiDiscrete([move, delta])` to
    `Discrete(move*delta)`.

    Reward modes:
    - `ppo_compatible`: identical to PPO-shaped reward.
    - `profit_oriented_v1`: terminal-profit reward without early/probing bonuses.
      `phase_aware_v1` is kept as a backward-compatible alias.
      Optionally supports per-step no-deal penalties to discourage stalling.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        catalog_path: str | Path,
        persona_config_path: str | Path,
        persona_bank_path: str | Path | None = None,
        persona_bank_split: str = "train",
        price_bin_count: int = 301,
        price_step_usd: float = 20.0,
        reward_scale_usd: float = 1000.0,
        no_deal_penalty: float = -0.15,
        profit_target_usd: float = 1500.0,
        low_profit_penalty: float = -0.3,
        soft_shortfall_penalty_coeff: float = 0.0,
        no_deal_requires_positive_margin: bool = True,
        invalid_accept_penalty: float = -0.2,
        initial_offer_markup: float = 1.8,
        clip_enabled: bool = False,
        clip_semantic_path: str | Path | None = None,  # clip semantic scores
        clip_legacy_proxy_enabled: bool = True,
        reward_mode: str = "profit_oriented_v1",
        step_no_deal_penalty: float = 0.0,
        step_no_deal_penalty_start_round: int = 3,
        grace_rounds_no_deal_penalty: int = 3,
        early_deal_round_cutoff: int = 3,
        early_deal_bonus: float = 0.0,
        delay_penalty_start_round: int = 4,
        delay_penalty_per_round: float = 0.0,
        probe_bonus_round_cutoff: int = 2,
        probe_bonus: float = 0.0,
    ) -> None:
        """Initializes the flat-action environment.

        Args:
            catalog_path: Customization catalog YAML path.
            persona_config_path: Persona config YAML path.
            persona_bank_path: Optional persona-bank JSONL path.
            persona_bank_split: Persona-bank split (`train/val/test`).
            price_bin_count: Number of delta bins.
            price_step_usd: USD step per delta bin.
            reward_scale_usd: Profit scaling divisor for terminal deal rewards.
            no_deal_penalty: No-deal terminal reward.
            profit_target_usd: Minimum target profit in USD for successful deals.
            low_profit_penalty: Extra penalty when terminal deal profit is below target.
            soft_shortfall_penalty_coeff: Continuous shortfall coefficient applied
                to `(profit_target_usd - profit_usd)_+ / reward_scale_usd` on deal.
            no_deal_requires_positive_margin: If true, no-deal penalty is applied
                only when there was positive feasible margin.
            invalid_accept_penalty: Penalty for invalid accept attempts.
            initial_offer_markup: Initial offer anchor multiplier.
            clip_enabled: Whether to append CLIP semantics to observations.
            clip_semantic_path: JSON path of offline CLIP semantics artifact.
            clip_legacy_proxy_enabled: Whether to keep proxy aesthetic scalar in
                base observation slot when CLIP is enabled.
            reward_mode: Reward mode token.
            step_no_deal_penalty: Penalty applied on every non-terminal step
                where a deal has not yet been reached.
            step_no_deal_penalty_start_round: First round index (1-based) from
                which non-terminal no-deal step penalty becomes active.
            grace_rounds_no_deal_penalty: Legacy argument retained for backward compatibility.
            early_deal_round_cutoff: Legacy argument retained for backward compatibility.
            early_deal_bonus: Legacy argument retained for backward compatibility.
            delay_penalty_start_round: Legacy argument retained for backward compatibility.
            delay_penalty_per_round: Legacy argument retained for backward compatibility.
            probe_bonus_round_cutoff: Legacy argument retained for backward compatibility.
            probe_bonus: Legacy argument retained for backward compatibility.
        """

        if not GYMNASIUM_AVAILABLE:
            raise RuntimeError("Gymnasium is required for DreamerDiscretePricingEnv.")
        if reward_mode not in {"ppo_compatible", "phase_aware_v1", "profit_oriented_v1"}:
            raise ValueError("`reward_mode` must be one of: `ppo_compatible`, `phase_aware_v1`, `profit_oriented_v1`.")
        if float(profit_target_usd) < 0.0:
            raise ValueError("`profit_target_usd` must be >= 0.")
        if float(low_profit_penalty) > 0.0:
            raise ValueError("`low_profit_penalty` must be <= 0.")
        if float(soft_shortfall_penalty_coeff) < 0.0:
            raise ValueError("`soft_shortfall_penalty_coeff` must be >= 0.")
        if int(grace_rounds_no_deal_penalty) < 0:
            raise ValueError("`grace_rounds_no_deal_penalty` must be >= 0.")
        if float(step_no_deal_penalty) > 0.0:
            raise ValueError("`step_no_deal_penalty` must be <= 0.")
        if int(step_no_deal_penalty_start_round) < 1:
            raise ValueError("`step_no_deal_penalty_start_round` must be >= 1.")
        if int(early_deal_round_cutoff) < 1:
            raise ValueError("`early_deal_round_cutoff` must be >= 1.")
        if float(early_deal_bonus) < 0.0:
            raise ValueError("`early_deal_bonus` must be >= 0.")
        if int(delay_penalty_start_round) < 1:
            raise ValueError("`delay_penalty_start_round` must be >= 1.")
        if float(delay_penalty_per_round) < 0.0:
            raise ValueError("`delay_penalty_per_round` must be >= 0.")
        if int(probe_bonus_round_cutoff) < 1:
            raise ValueError("`probe_bonus_round_cutoff` must be >= 1.")
        if float(probe_bonus) < 0.0:
            raise ValueError("`probe_bonus` must be >= 0.")

        super().__init__()
        self._base_env = PPOPricingEnv(
            catalog_path=catalog_path,
            persona_config_path=persona_config_path,
            persona_bank_path=persona_bank_path,
            persona_bank_split=persona_bank_split,
            price_bin_count=price_bin_count,
            price_step_usd=price_step_usd,
            reward_scale_usd=reward_scale_usd,
            no_deal_penalty=no_deal_penalty,
            profit_target_usd=profit_target_usd,
            low_profit_penalty=low_profit_penalty,
            no_deal_requires_positive_margin=no_deal_requires_positive_margin,
            invalid_accept_penalty=invalid_accept_penalty,
            initial_offer_markup=initial_offer_markup,
            clip_enabled=clip_enabled,
            clip_semantic_path=clip_semantic_path,
            clip_legacy_proxy_enabled=clip_legacy_proxy_enabled,
        )
        self._codec = DreamerActionCodec(
            move_count=int(self._base_env.action_space.nvec[0]),
            delta_bin_count=int(self._base_env.action_space.nvec[1]),
        )
        self._reward_mode = str(reward_mode)
        self._reward_scale_usd = float(reward_scale_usd)
        self._no_deal_penalty = float(no_deal_penalty)
        self._step_no_deal_penalty = float(step_no_deal_penalty)
        self._step_no_deal_penalty_start_round = int(step_no_deal_penalty_start_round)
        self._profit_target_usd = float(profit_target_usd)
        self._low_profit_penalty = float(low_profit_penalty)
        self._soft_shortfall_penalty_coeff = float(soft_shortfall_penalty_coeff)
        self._no_deal_requires_positive_margin = bool(no_deal_requires_positive_margin)
        self._invalid_accept_penalty = float(invalid_accept_penalty)
        self._grace_rounds_no_deal_penalty = int(grace_rounds_no_deal_penalty)
        self._early_deal_round_cutoff = int(early_deal_round_cutoff)
        self._early_deal_bonus = float(early_deal_bonus)
        self._delay_penalty_start_round = int(delay_penalty_start_round)
        self._delay_penalty_per_round = float(delay_penalty_per_round)
        self._probe_bonus_round_cutoff = int(probe_bonus_round_cutoff)
        self._probe_bonus = float(probe_bonus)
        self.observation_space = self._base_env.observation_space
        self.action_space = spaces.Discrete(self._codec.action_size)

    @property
    def codec(self) -> DreamerActionCodec:
        """Returns action codec used by this environment."""

        return self._codec

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets environment.

        Args:
            seed: Optional random seed.
            options: Optional reset options.

        Returns:
            Tuple of observation and info dictionary.
        """

        return self._base_env.reset(seed=seed, options=options)

    def step(self, action: int | np.integer | np.ndarray | Mapping[str, Any]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Applies one flattened action.

        Args:
            action: Flat discrete action id.

        Returns:
            Tuple `(obs, reward, terminated, truncated, info)`.
        """

        action_idx = self._coerce_action_index(action)
        move_idx, delta_idx = self._codec.unflatten(action_idx)
        base_action = np.array([move_idx, delta_idx], dtype=np.int64)
        obs, base_reward, terminated, truncated, info = self._base_env.step(base_action)

        if self._reward_mode == "ppo_compatible":
            info["dreamer_reward_mode"] = "ppo_compatible"
            info["dreamer_reward_shaped"] = float(base_reward)
            return obs, float(base_reward), bool(terminated), bool(truncated), info

        shaped_reward = self._shape_profit_oriented_reward(
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info,
        )
        if self._reward_mode == "phase_aware_v1":
            info["dreamer_reward_mode"] = "phase_aware_v1"
        else:
            info["dreamer_reward_mode"] = "profit_oriented_v1"
        info["dreamer_reward_shaped"] = float(shaped_reward)
        return obs, float(shaped_reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        """Closes underlying environment."""

        self._base_env.close()

    def _coerce_action_index(self, action: int | np.integer | np.ndarray | Mapping[str, Any]) -> int:
        """Converts action payload into one integer action id.

        Args:
            action: Input action payload.

        Returns:
            Parsed flat action index.
        """

        if isinstance(action, Mapping):
            if "action" not in action:
                raise ValueError("Mapping action must contain key `action`.")
            raw = action["action"]
        else:
            raw = action

        if isinstance(raw, np.ndarray):
            flat = np.asarray(raw).reshape(-1)
            if flat.size != 1:
                raise ValueError("Flattened action array must contain exactly one scalar.")
            action_idx = int(flat[0])
        else:
            action_idx = int(raw)

        if action_idx < 0 or action_idx >= self._codec.action_size:
            raise ValueError(f"Invalid flattened action index: {action_idx}.")
        return action_idx

    def _shape_profit_oriented_reward(
        self,
        *,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> float:
        """Computes Dreamer-only profit-oriented reward."""

        raw_observation = info.get("raw_observation", {})
        episode_metrics = info.get("episode_metrics", {})
        is_terminal = bool(terminated or truncated)

        if is_terminal:
            deal_reached = bool(episode_metrics.get("deal_reached", False))
            if deal_reached:
                profit_usd = float(episode_metrics.get("profit_usd", 0.0))
                reward = profit_usd / self._reward_scale_usd
                low_profit_applied = profit_usd < self._profit_target_usd
                if low_profit_applied:
                    reward += self._low_profit_penalty
                shortfall_usd = max(0.0, self._profit_target_usd - profit_usd)
                soft_shortfall_penalty = (
                    self._soft_shortfall_penalty_coeff * (shortfall_usd / self._reward_scale_usd)
                )
                reward -= soft_shortfall_penalty
                info["dreamer_profit_component"] = float(profit_usd / self._reward_scale_usd)
                info["dreamer_low_profit_penalty"] = float(self._low_profit_penalty if low_profit_applied else 0.0)
                info["dreamer_soft_shortfall_penalty_applied"] = float(soft_shortfall_penalty)
                info["dreamer_no_deal_penalty_applied"] = 0.0
                info["dreamer_step_no_deal_penalty_applied"] = 0.0
                return float(reward)

            counter_offer = raw_observation.get("last_consumer_offer_usd")
            total_cost = float(raw_observation.get("total_customization_cost_usd", 0.0))
            estimated_max_feasible_profit = None
            if counter_offer is not None:
                estimated_max_feasible_profit = float(counter_offer) - total_cost
            should_penalize = (not self._no_deal_requires_positive_margin) or (
                estimated_max_feasible_profit is not None and estimated_max_feasible_profit > 0.0
            )
            reward = float(self._no_deal_penalty if should_penalize else 0.0)
            info["dreamer_profit_component"] = 0.0
            info["dreamer_low_profit_penalty"] = 0.0
            info["dreamer_soft_shortfall_penalty_applied"] = 0.0
            info["dreamer_no_deal_penalty_applied"] = float(reward)
            info["dreamer_step_no_deal_penalty_applied"] = 0.0
            info["dreamer_estimated_max_feasible_profit_usd"] = (
                None if estimated_max_feasible_profit is None else float(estimated_max_feasible_profit)
            )
            return float(reward)

        # `round_idx` in raw observation points to next round after transition.
        # Convert to the action round index so start-round gating is intuitive.
        next_round_idx = int(raw_observation.get("round_idx", 1))
        action_round_idx = max(1, next_round_idx - 1)
        step_penalty_applied = (
            float(self._step_no_deal_penalty)
            if action_round_idx >= int(self._step_no_deal_penalty_start_round)
            else 0.0
        )
        reward = float(step_penalty_applied)
        if str(raw_observation.get("last_consumer_response", "")) == "invalid_accept":
            reward += self._invalid_accept_penalty
        info["dreamer_profit_component"] = 0.0
        info["dreamer_low_profit_penalty"] = 0.0
        info["dreamer_soft_shortfall_penalty_applied"] = 0.0
        info["dreamer_no_deal_penalty_applied"] = 0.0
        info["dreamer_step_no_deal_penalty_applied"] = float(step_penalty_applied)
        info["dreamer_step_no_deal_penalty_start_round"] = int(self._step_no_deal_penalty_start_round)
        info["dreamer_step_no_deal_penalty_action_round_idx"] = int(action_round_idx)
        return float(reward)
