"""PPO-specific Gym environment wrapper for pricing negotiation.

This module adapts the generic pricing Gym wrapper to a PPO-friendly interface:
- discrete action space with move + price-delta bins
- profit-dominant reward shaping
- explicit metadata for debugging and experiment reproducibility
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from pricing_env.gym_wrapper import GYMNASIUM_AVAILABLE, PricingNegotiationGymWrapper

if GYMNASIUM_AVAILABLE:
    import gymnasium as gym
    from gymnasium import spaces
else:  # pragma: no cover - optional dependency path
    gym = object  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]


MOVE_TOKENS = ("offer", "accept", "walkaway")


class PPOPricingEnv(gym.Env if GYMNASIUM_AVAILABLE else object):
    """PPO-facing environment with discrete delta actions and profit reward.

    Design goals:
    - Keep environment dynamics unchanged (delegates all transitions to `PricingNegotiationGymWrapper`).
    - Expose a stable, fully discrete action space for PPO.
    - Shape rewards around end-of-episode profit, matching the project objective.

    Action encoding:
    - `action[0]`: move index (`0=offer`, `1=accept`, `2=walkaway`)
    - `action[1]`: price-delta bin index in `[0, price_bin_count - 1]`

    Reward shaping:
    - Deal reached: `profit_usd / reward_scale_usd`
    - Low-profit deal: additional `low_profit_penalty` if `profit_usd < profit_target_usd`
    - No-deal terminal states (walkaway/timeout):
      - `no_deal_penalty` when there was positive feasible margin
      - `0.0` when feasible margin was non-positive
    - Non-terminal no-deal steps: `step_no_deal_penalty`
      (applied only when round index reaches configured start round)
    - Invalid accept while episode continues: `invalid_accept_penalty`
    - Non-terminal regular steps: `step_no_deal_penalty`
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
        step_no_deal_penalty: float = 0.0,
        step_no_deal_penalty_start_round: int = 3,
        profit_target_usd: float = 1500.0,
        low_profit_penalty: float = -0.3,
        no_deal_requires_positive_margin: bool = True,
        invalid_accept_penalty: float = -0.2,
        initial_offer_markup: float = 1.8,
        clip_enabled: bool = False,
        clip_semantic_path: str | Path | None = None,
        clip_legacy_proxy_enabled: bool = True,
    ) -> None:
        """Initializes PPO environment adapter.

        Args:
            catalog_path: Customization catalog YAML.
            persona_config_path: Persona config YAML.
            persona_bank_path: Optional offline persona bank JSONL path.
            persona_bank_split: Split label used by persona bank (`train`/`val`/`test`).
            price_bin_count: Number of discrete delta bins. Must be odd and >= 3.
            price_step_usd: USD step between adjacent delta bins.
            reward_scale_usd: Profit normalization divisor for terminal deal rewards.
            no_deal_penalty: Terminal reward for no-deal outcomes.
            step_no_deal_penalty: Penalty applied on every non-terminal step
                where no deal has been reached yet.
            step_no_deal_penalty_start_round: First round index (1-based) from
                which non-terminal no-deal step penalty becomes active.
            profit_target_usd: Minimum target profit in USD for successful deals.
            low_profit_penalty: Extra penalty when terminal deal profit is below target.
            no_deal_requires_positive_margin: If true, no-deal penalty is applied only
                when the current step still had positive feasible margin.
            invalid_accept_penalty: Penalty for invalid accept attempts.
            initial_offer_markup: Initial anchor ratio applied to configuration MSRP delta.
            clip_enabled: Whether to append CLIP semantics to observations.
            clip_semantic_path: JSON path of offline CLIP semantics artifact.
            clip_legacy_proxy_enabled: Whether to keep proxy aesthetic scalar in base
                observation slot when CLIP is enabled.
        """

        if not GYMNASIUM_AVAILABLE:
            raise RuntimeError("Gymnasium is required for PPOPricingEnv.")
        if price_bin_count < 3 or price_bin_count % 2 == 0:
            raise ValueError("`price_bin_count` must be an odd integer >= 3.")
        if price_step_usd <= 0.0:
            raise ValueError("`price_step_usd` must be > 0.")
        if reward_scale_usd <= 0.0:
            raise ValueError("`reward_scale_usd` must be > 0.")
        if profit_target_usd < 0.0:
            raise ValueError("`profit_target_usd` must be >= 0.")
        if low_profit_penalty > 0.0:
            raise ValueError("`low_profit_penalty` must be <= 0.")
        if step_no_deal_penalty > 0.0:
            raise ValueError("`step_no_deal_penalty` must be <= 0.")
        if step_no_deal_penalty_start_round < 1:
            raise ValueError("`step_no_deal_penalty_start_round` must be >= 1.")
        if initial_offer_markup <= 0.0:
            raise ValueError("`initial_offer_markup` must be > 0.")

        super().__init__()
        self._base_env = PricingNegotiationGymWrapper(
            catalog_path=catalog_path,
            persona_config_path=persona_config_path,
            persona_bank_path=persona_bank_path,
            persona_bank_split=persona_bank_split,
            clip_enabled=clip_enabled,
            clip_semantic_path=clip_semantic_path,
            clip_legacy_proxy_enabled=clip_legacy_proxy_enabled,
        )
        self.observation_space = self._base_env.observation_space
        self.action_space = spaces.MultiDiscrete([len(MOVE_TOKENS), int(price_bin_count)])

        self._price_bin_count = int(price_bin_count)
        self._delta_center = int(price_bin_count // 2)
        self._price_step_usd = float(price_step_usd)
        self._reward_scale_usd = float(reward_scale_usd)
        self._no_deal_penalty = float(no_deal_penalty)
        self._step_no_deal_penalty = float(step_no_deal_penalty)
        self._step_no_deal_penalty_start_round = int(step_no_deal_penalty_start_round)
        self._profit_target_usd = float(profit_target_usd)
        self._low_profit_penalty = float(low_profit_penalty)
        self._no_deal_requires_positive_margin = bool(no_deal_requires_positive_margin)
        self._invalid_accept_penalty = float(invalid_accept_penalty)
        self._initial_offer_markup = float(initial_offer_markup)

        self._offer_min = float(self._base_env.action_space["price"].low[0])
        self._offer_max = float(self._base_env.action_space["price"].high[0])
        self._reference_price_usd = self._offer_min

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets environment and initializes offer reference anchor.

        Args:
            seed: Optional random seed.
            options: Optional underlying environment reset options.

        Returns:
            Encoded observation and info dictionary.
        """

        obs, info = self._base_env.reset(seed=seed, options=options)
        raw_obs = info.get("raw_observation", {})
        total_msrp_delta = float(
            raw_obs.get(
                "total_msrp_delta_usd",
                raw_obs.get("total_customization_cost_usd", 0.0),
            )
        )
        initial_anchor = total_msrp_delta * self._initial_offer_markup
        self._reference_price_usd = self._clip_price(initial_anchor)
        info["ppo_reference_price_usd"] = self._reference_price_usd
        return obs, info

    def step(self, action: Mapping[str, Any] | np.ndarray | list | tuple) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Applies one PPO action and returns shaped reward.

        Args:
            action: Action in `MultiDiscrete` form.

        Returns:
            Tuple `(obs, shaped_reward, terminated, truncated, info)`.
        """

        move_idx, delta_idx = self._decode_action(action)
        move_token = MOVE_TOKENS[move_idx]
        delta_steps = int(delta_idx) - self._delta_center
        proposed_price = self._clip_price(self._reference_price_usd + delta_steps * self._price_step_usd)

        wrapped_action: Dict[str, Any] = {
            "move": move_idx,
            "price": [proposed_price if move_token == "offer" else 0.0],
        }
        obs, env_reward, terminated, truncated, info = self._base_env.step(wrapped_action)
        raw_obs = info.get("raw_observation", {})
        shaped_reward = self._shape_reward(
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info,
            raw_observation=raw_obs,
        )

        # Maintain a rolling anchor so delta actions reflect negotiation context.
        self._update_reference_price(raw_observation=raw_obs)

        info["ppo_move"] = move_token
        info["ppo_delta_steps"] = delta_steps
        info["ppo_offer_price_usd"] = proposed_price if move_token == "offer" else None
        info["ppo_reference_price_usd"] = self._reference_price_usd
        info["ppo_reward_raw_env"] = float(env_reward)
        info["ppo_reward_shaped"] = float(shaped_reward)
        return obs, float(shaped_reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        """Closes the underlying wrapper."""

        self._base_env.close()

    def _decode_action(self, action: Mapping[str, Any] | np.ndarray | list | tuple) -> Tuple[int, int]:
        """Decodes action into `(move_idx, delta_idx)`.

        Args:
            action: Candidate action object.

        Returns:
            Parsed move and delta bin indices.
        """

        if isinstance(action, Mapping):
            move_idx = int(action["move"])
            delta_idx = int(action["delta"])
        else:
            action_array = np.asarray(action, dtype=np.int64).reshape(-1)
            if action_array.size != 2:
                raise ValueError("PPO action must contain exactly two integers: [move_idx, delta_idx].")
            move_idx = int(action_array[0])
            delta_idx = int(action_array[1])

        if move_idx < 0 or move_idx >= len(MOVE_TOKENS):
            raise ValueError(f"Invalid move index: {move_idx}.")
        if delta_idx < 0 or delta_idx >= self._price_bin_count:
            raise ValueError(f"Invalid delta index: {delta_idx}.")
        return move_idx, delta_idx

    def _shape_reward(
        self,
        *,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
        raw_observation: Dict[str, Any],
    ) -> float:
        """Computes PPO-specific reward from episode outcomes.

        Args:
            terminated: Gymnasium terminated flag.
            truncated: Gymnasium truncated flag.
            info: Step info dictionary.
            raw_observation: Raw observation payload.

        Returns:
            Shaped reward scalar.
        """

        if terminated or truncated:
            episode_metrics = info.get("episode_metrics", {})
            if bool(episode_metrics.get("deal_reached", False)):
                profit_usd = float(episode_metrics.get("profit_usd", 0.0))
                reward = profit_usd / self._reward_scale_usd
                low_profit_applied = profit_usd < self._profit_target_usd
                if low_profit_applied:
                    reward += self._low_profit_penalty
                info["ppo_profit_component"] = float(profit_usd / self._reward_scale_usd)
                info["ppo_low_profit_penalty"] = float(self._low_profit_penalty if low_profit_applied else 0.0)
                info["ppo_no_deal_penalty_applied"] = 0.0
                info["ppo_step_no_deal_penalty_applied"] = 0.0
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
            info["ppo_profit_component"] = 0.0
            info["ppo_low_profit_penalty"] = 0.0
            info["ppo_no_deal_penalty_applied"] = float(reward)
            info["ppo_step_no_deal_penalty_applied"] = 0.0
            info["ppo_estimated_max_feasible_profit_usd"] = (
                None if estimated_max_feasible_profit is None else float(estimated_max_feasible_profit)
            )
            return reward

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
        info["ppo_step_no_deal_penalty_applied"] = float(step_penalty_applied)
        info["ppo_step_no_deal_penalty_start_round"] = int(self._step_no_deal_penalty_start_round)
        info["ppo_step_no_deal_penalty_action_round_idx"] = int(action_round_idx)
        return float(reward)

    def _update_reference_price(self, *, raw_observation: Dict[str, Any]) -> None:
        """Updates rolling reference price from latest negotiation state.

        Args:
            raw_observation: Raw observation from underlying environment.
        """

        counter = raw_observation.get("last_consumer_offer_usd")
        if counter is not None:
            self._reference_price_usd = self._clip_price(float(counter))
            return

        last_offer = raw_observation.get("last_agent_offer_usd")
        if last_offer is not None:
            self._reference_price_usd = self._clip_price(float(last_offer))

    def _clip_price(self, value: float) -> float:
        """Clips price into environment action bounds.

        Args:
            value: Candidate USD price.

        Returns:
            Clipped USD price.
        """

        return float(np.clip(float(value), self._offer_min, self._offer_max))
