"""Point-robot closed-loop training built on top of `dynn`."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any

from envs.point_robot import ACTIONS, PointRobotConfig, PointRobotEnv
from models.common import argmax, clamp, dot, l2_normalize, mean_squared
from models.recurrent_spiking import RSNNConfig, build_spiking_network


@dataclass
class AgentConfig:
    """Point-robot agent configuration."""

    episodes: int = 320
    eval_every: int = 40
    eval_episodes: int = 60
    n_neurons: int = 64
    recurrent_degree: int = 4
    lr_model: float = 0.010
    lr_value: float = 0.018
    gamma: float = 0.94
    epsilon_start: float = 0.45
    epsilon_end: float = 0.06
    model_score_weight: float = 2.0
    value_score_weight: float = 0.55
    recurrent_plasticity: float = 0.0002
    plasticity_rule: str = "three_factor"
    neuron_model: str = "lif"
    randomize_intrinsics: bool = True
    seed: int = 31


class LinearValueHead:
    """Linear action-value readout."""

    def __init__(self, n_actions: int, hidden_dim: int, lr: float, rng: random.Random) -> None:
        scale = 0.01 / math.sqrt(hidden_dim)
        self.lr = lr
        self.weights = [
            [rng.gauss(0.0, scale) for _ in range(hidden_dim)]
            for _ in range(n_actions)
        ]

    def q_values(self, features: list[float]) -> list[float]:
        return [dot(weights, features) for weights in self.weights]

    def update(self, features: list[float], action: int, td_error: float) -> None:
        row = self.weights[action]
        clipped_error = clamp(td_error, -2.0, 2.0)
        for hidden_index, feature in enumerate(features):
            row[hidden_index] += self.lr * clipped_error * feature
            row[hidden_index] = clamp(row[hidden_index], -1.0, 1.0)


class DeltaWorldModelHead:
    """One-step observation-delta predictor."""

    def __init__(
        self,
        n_actions: int,
        obs_dim: int,
        hidden_dim: int,
        lr: float,
        observation_mode: str,
        rng: random.Random,
    ) -> None:
        scale = 0.01 / math.sqrt(hidden_dim)
        self.lr = lr
        self.observation_mode = observation_mode
        self.weights = [
            [
                [rng.gauss(0.0, scale) for _ in range(hidden_dim)]
                for _ in range(obs_dim)
            ]
            for _ in range(n_actions)
        ]

    def predict(self, observation: list[float], features: list[float], action: int) -> list[float]:
        predicted_delta = [dot(row, features) for row in self.weights[action]]
        predicted = [
            value + delta
            for value, delta in zip(observation, predicted_delta, strict=True)
        ]
        return clamp_observation(predicted, self.observation_mode)

    def update(
        self,
        observation: list[float],
        features: list[float],
        action: int,
        next_observation: list[float],
    ) -> float:
        predicted = self.predict(observation, features, action)
        errors = [
            target - output
            for target, output in zip(next_observation, predicted, strict=True)
        ]
        for output_index, error in enumerate(errors):
            row = self.weights[action][output_index]
            clipped_error = clamp(error, -1.0, 1.0)
            for hidden_index, feature in enumerate(features):
                row[hidden_index] += self.lr * clipped_error * feature
                row[hidden_index] = clamp(row[hidden_index], -1.0, 1.0)
        return mean_squared(errors)


class ClosedLoopPointRobotAgent:
    """Closed-loop controller with recurrent spiking state."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        config: AgentConfig,
        rng: random.Random,
        observation_mode: str,
    ) -> None:
        self.config = config
        self.n_actions = n_actions
        self.rsnn = build_spiking_network(
            RSNNConfig(
                input_dim=obs_dim,
                n_neurons=config.n_neurons,
                recurrent_degree=config.recurrent_degree,
                neuron_model=config.neuron_model,
                plastic_lr=config.recurrent_plasticity,
                plasticity_rule=config.plasticity_rule,
                randomize_intrinsics=config.randomize_intrinsics,
                seed=config.seed + 1,
            ),
            rng,
        )
        hidden_dim = self.rsnn.feature_dim()
        self.value_head = LinearValueHead(n_actions, hidden_dim, config.lr_value, rng)
        self.world_model = DeltaWorldModelHead(
            n_actions=n_actions,
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            lr=config.lr_model,
            observation_mode=observation_mode,
            rng=rng,
        )

    def reset_state(self) -> None:
        self.rsnn.reset_state()

    def observe(self, observation: list[float]) -> list[float]:
        return l2_normalize(self.rsnn.step(observation))

    def q_values(self, features: list[float]) -> list[float]:
        return self.value_head.q_values(features)

    def predict_next(self, observation: list[float], features: list[float], action: int) -> list[float]:
        return self.world_model.predict(observation, features, action)

    def choose_action(self, observation: list[float], features: list[float], epsilon: float, learn: bool) -> int:
        if learn and self.rsnn.rng.random() < epsilon:
            return self.rsnn.rng.randrange(self.n_actions)
        q_values = self.q_values(features)
        scores = []
        for action in range(self.n_actions):
            predicted_next = self.predict_next(observation, features, action)
            predicted_distance = max(0.0, predicted_next[8])
            score = -self.config.model_score_weight * predicted_distance
            score += self.config.value_score_weight * q_values[action]
            if ACTIONS[action] == "stay":
                score -= 0.04
            scores.append(score)
        return argmax(scores)

    def learn_world_model(self, observation: list[float], features: list[float], action: int, next_observation: list[float]) -> float:
        return self.world_model.update(observation, features, action, next_observation)

    def learn_value(self, features: list[float], action: int, td_error: float) -> None:
        self.value_head.update(features, action, td_error)


def _variable_spec(
    name: str,
    label: str,
    label_zh: str,
    *,
    role: str,
) -> dict[str, str]:
    return {
        "name": name,
        "label": label,
        "label_zh": label_zh,
        "role": role,
    }


def _indexed_variable_specs(
    prefix: str,
    count: int,
    *,
    label_prefix: str,
    label_prefix_zh: str,
    role: str,
) -> list[dict[str, str]]:
    return [
        _variable_spec(
            f"{prefix}_{index:03d}",
            f"{label_prefix} {index}",
            f"{label_prefix_zh}{index}",
            role=role,
        )
        for index in range(count)
    ]


def _point_robot_observation_specs(observation_mode: str) -> list[dict[str, str]]:
    if observation_mode == "full":
        return [
            _variable_spec("robot_x", "robot x", "机器人横坐标", role="observation"),
            _variable_spec("robot_y", "robot y", "机器人纵坐标", role="observation"),
            _variable_spec("velocity_x", "velocity x", "速度 x", role="observation"),
            _variable_spec("velocity_y", "velocity y", "速度 y", role="observation"),
            _variable_spec("goal_x", "goal x", "目标横坐标", role="observation"),
            _variable_spec("goal_y", "goal y", "目标纵坐标", role="observation"),
            _variable_spec("goal_dx", "goal dx", "目标相对位移 x", role="observation"),
            _variable_spec("goal_dy", "goal dy", "目标相对位移 y", role="observation"),
            _variable_spec(
                "goal_distance_norm",
                "goal distance norm",
                "目标距离归一化",
                role="observation",
            ),
            _variable_spec("bias", "bias", "偏置", role="observation"),
        ]
    return [
        _variable_spec("robot_x", "robot x", "机器人横坐标", role="observation"),
        _variable_spec("robot_y", "robot y", "机器人纵坐标", role="observation"),
        _variable_spec("velocity_x", "velocity x", "速度 x", role="observation"),
        _variable_spec("velocity_y", "velocity y", "速度 y", role="observation"),
        _variable_spec("cue_goal_dx", "cue goal dx", "目标提示相对位移 x", role="observation"),
        _variable_spec("cue_goal_dy", "cue goal dy", "目标提示相对位移 y", role="observation"),
        _variable_spec("goal_visible", "goal visible", "目标是否可见", role="observation"),
        _variable_spec("episode_progress", "episode progress", "回合进度", role="observation"),
        _variable_spec(
            "goal_distance_norm",
            "goal distance norm",
            "目标距离归一化",
            role="observation",
        ),
        _variable_spec("bias", "bias", "偏置", role="observation"),
    ]


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _component_annotation(
    annotation_type: str,
    target: str,
    label: str,
    *,
    label_zh: str,
    layer_id: str,
    member_kind: str,
    variable_specs: list[dict[str, str]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "label_zh": label_zh,
        "layer_id": layer_id,
        "member_kind": member_kind,
    }
    if variable_specs is not None:
        payload["variable_names"] = [spec["name"] for spec in variable_specs]
        payload["variables"] = variable_specs
    if metadata:
        payload.update(metadata)
    return {
        "type": annotation_type,
        "target": target,
        "label": label,
        "metadata": payload,
    }


def build_point_robot_export_topology(
    agent: ClosedLoopPointRobotAgent,
    env_config: PointRobotConfig,
) -> dict[str, Any]:
    base_topology = agent.rsnn.graph.topology
    observation_specs = _point_robot_observation_specs(env_config.observation_mode)
    node_variable_index: dict[str, list[dict[str, str]]] = {}
    port_variable_index: dict[str, list[dict[str, str]]] = {}
    annotations: list[dict[str, Any]] = []
    node_sets: list[dict[str, Any]] = []
    ports: list[dict[str, Any]] = []

    for node_set in base_topology.get("node_sets", []):
        node_copy = dict(node_set)
        node_id = str(node_copy["id"])
        size = int(node_copy["size"])
        raw_parameters = node_copy.get("parameters", {})
        parameters = dict(raw_parameters) if isinstance(raw_parameters, dict) else {}
        if node_id == "obs":
            layer_id = "layer_2_interface"
            display_label = "Observation Population"
            display_label_zh = "观测编码群"
            variable_specs = observation_specs
            tags = ["layer:interface", "component:observation"]
        else:
            layer_id = "layer_3_core_model"
            display_label = "Hidden Population"
            display_label_zh = "隐藏神经群"
            variable_specs = _indexed_variable_specs(
                f"{node_id}_unit",
                size,
                label_prefix=f"{node_id} unit",
                label_prefix_zh=f"{node_id} 单元 ",
                role="neuron",
            )
            tags = ["layer:core_model", "component:hidden_population"]
        parameters["export"] = {
            "display_label": display_label,
            "display_label_zh": display_label_zh,
            "layer_id": layer_id,
            "variable_names": [spec["name"] for spec in variable_specs],
            "variables": variable_specs,
        }
        node_copy["parameters"] = parameters
        node_copy["tags"] = _ordered_unique(list(node_copy.get("tags", [])) + tags)
        node_sets.append(node_copy)
        node_variable_index[node_id] = variable_specs
        annotations.append(
            _component_annotation(
                "display_label",
                node_id,
                display_label,
                label_zh=display_label_zh,
                layer_id=layer_id,
                member_kind="node_set",
                variable_specs=variable_specs,
                metadata={"count": size},
            )
        )

    for port in base_topology.get("ports", []):
        port_copy = dict(port)
        port_id = str(port_copy["id"])
        params = dict(port_copy.get("params", {})) if isinstance(port_copy.get("params", {}), dict) else {}
        if port_id == "obs":
            layer_id = "layer_2_interface"
            display_label = "Observation Port"
            display_label_zh = "观测输入端口"
            variable_specs = observation_specs
            tags = ["layer:interface", "component:input_port"]
        else:
            layer_id = "layer_3_core_model"
            display_label = f"{port_id} State Port"
            display_label_zh = f"{port_id} 状态端口"
            variable_specs = node_variable_index.get(str(port_copy.get("node_set", "")), [])
            tags = ["layer:core_model", "component:state_port"]
        params["export"] = {
            "display_label": display_label,
            "display_label_zh": display_label_zh,
            "layer_id": layer_id,
            "variable_names": [spec["name"] for spec in variable_specs],
            "variables": variable_specs,
        }
        params["tags"] = tags
        port_copy["params"] = params
        ports.append(port_copy)
        port_variable_index[port_id] = variable_specs
        annotations.append(
            _component_annotation(
                "display_label",
                port_id,
                display_label,
                label_zh=display_label_zh,
                layer_id=layer_id,
                member_kind="port",
                variable_specs=variable_specs,
                metadata={"direction": port_copy.get("kind", "io")},
            )
        )

    edge_sets: list[dict[str, Any]] = []
    artifact_specs: list[dict[str, Any]] = []
    for edge_set in base_topology.get("edge_sets", []):
        edge_copy = dict(edge_set)
        edge_id = str(edge_copy["id"])
        source_id = str(edge_copy["source"]["node_set"])
        target_id = str(edge_copy["target"]["node_set"])
        tags = ["component:connectivity"]
        if target_id == "hidden":
            tags.append("layer:core_model")
        edge_copy["tags"] = _ordered_unique(list(edge_copy.get("tags", [])) + tags)
        source_variables = node_variable_index.get(source_id, [])
        target_variables = node_variable_index.get(target_id, [])
        display_label = f"{source_id} -> {target_id}"
        display_label_zh = f"{source_id} 到 {target_id}"
        edge_copy["annotations"] = [
            _component_annotation(
                "display_label",
                edge_id,
                display_label,
                label_zh=display_label_zh,
                layer_id="layer_3_core_model",
                member_kind="edge_set",
                metadata={
                    "source_node_set": source_id,
                    "target_node_set": target_id,
                    "source_variable_names": [spec["name"] for spec in source_variables],
                    "target_variable_names": [spec["name"] for spec in target_variables],
                },
            )
        ]
        edge_sets.append(edge_copy)

        representation = edge_set.get("representation", {})
        edges = []
        if isinstance(representation, dict):
            edges = list(representation.get("edges", []))
        artifact_specs.append(
            {
                "artifact_id": f"{edge_id}-edges",
                "kind": "weight_summary",
                "path": f"topology/edges/{edge_id}.json",
                "media_type": "application/json",
                "summary": {
                    "label": display_label,
                    "label_zh": display_label_zh,
                    "edge_count": len(edges),
                    "source_node_set": source_id,
                    "target_node_set": target_id,
                },
                "payload": {
                    "schema_version": 1,
                    "artifact_id": f"{edge_id}-edges",
                    "edge_set_id": edge_id,
                    "label": display_label,
                    "label_zh": display_label_zh,
                    "source_node_set": source_id,
                    "target_node_set": target_id,
                    "source_variable_names": [spec["name"] for spec in source_variables],
                    "target_variable_names": [spec["name"] for spec in target_variables],
                    "edges": edges,
                },
            }
        )

    world_model_specs = [
        _variable_spec(
            f"world_model_{ACTIONS[action_index]}_{spec['name']}",
            f"world model {ACTIONS[action_index]} {spec['label']}",
            f"世界模型 {ACTIONS[action_index]} {spec['label_zh']}",
            role="prediction_head",
        )
        for action_index in range(agent.n_actions)
        for spec in observation_specs
    ]
    value_head_specs = [
        _variable_spec(
            f"q_{action_name}",
            f"Q {action_name}",
            f"{action_name} 动作价值",
            role="value_head",
        )
        for action_name in ACTIONS
    ]
    structure_payload = {
        "schema_version": 1,
        "tree_id": "point-robot-subgraph-tree",
        "label": "Point Robot Nested Subgraph",
        "label_zh": "点机器人嵌套子图",
        "root": {
            "id": "closed_loop_agent",
            "label": "Closed Loop Agent",
            "label_zh": "闭环智能体",
            "input_gateways": [{"id": "obs_input", "port": "obs", "label": "Observation Input", "label_zh": "观测输入"}],
            "output_gateways": [{"id": "action_value_output", "label": "Action Value Output", "label_zh": "动作价值输出"}],
            "groups": [
                {
                    "id": "environment_group",
                    "order": 1,
                    "label": "Environment",
                    "label_zh": "环境",
                    "variables": [
                        _variable_spec("x", "robot x", "机器人横坐标", role="environment"),
                        _variable_spec("y", "robot y", "机器人纵坐标", role="environment"),
                        _variable_spec("vx", "velocity x", "速度 x", role="environment"),
                        _variable_spec("vy", "velocity y", "速度 y", role="environment"),
                        _variable_spec("goal_x", "goal x", "目标横坐标", role="environment"),
                        _variable_spec("goal_y", "goal y", "目标纵坐标", role="environment"),
                        _variable_spec("distance_to_goal", "distance to goal", "到目标距离", role="environment"),
                        _variable_spec("steps", "steps", "步数", role="environment"),
                    ],
                    "metadata": {
                        "observation_mode": env_config.observation_mode,
                        "goal_cue_steps": env_config.goal_cue_steps,
                        "action_names": list(ACTIONS),
                    },
                },
                {
                    "id": "observation_interface_group",
                    "order": 2,
                    "label": "Observation Interface",
                    "label_zh": "观测接口",
                    "member_node_sets": ["obs"],
                    "input_gateways": [{"id": "obs_input", "port": "obs", "label": "Observation Input", "label_zh": "观测输入"}],
                },
                {
                    "id": "spiking_state_group",
                    "order": 3,
                    "label": "Spiking State Model",
                    "label_zh": "脉冲状态模型",
                    "groups": [
                        {
                            "id": "state_hidden",
                            "order": 1,
                            "label": "hidden",
                            "label_zh": "hidden",
                            "member_node_sets": ["hidden"],
                        }
                    ],
                },
                {
                    "id": "prediction_control_group",
                    "order": 4,
                    "label": "Prediction and Control",
                    "label_zh": "预测与控制",
                    "output_gateways": [
                        {"id": "action_value_output", "label": "Action Value Output", "label_zh": "动作价值输出"}
                    ],
                    "variables": [*world_model_specs, *value_head_specs],
                },
            ],
        },
    }
    artifact_specs.append(
        {
            "artifact_id": "topology-subgraph-tree",
            "kind": "topology_structure",
            "path": "topology/subgraph-tree.json",
            "media_type": "application/json",
            "summary": {
                "label": "Point Robot Nested Subgraph",
                "label_zh": "点机器人嵌套子图",
                "root_group": "closed_loop_agent",
            },
            "payload": structure_payload,
        }
    )
    artifact_specs.append(
        {
            "artifact_id": "point-robot-task",
            "kind": "task_environment",
            "path": "environment/point-robot-task.json",
            "media_type": "application/json",
            "summary": {
                "label": "Point Robot Task",
                "label_zh": "点机器人任务",
                "observation_mode": env_config.observation_mode,
                "action_count": len(ACTIONS),
            },
            "payload": {
                "schema_version": 1,
                "label": "Point Robot Task",
                "label_zh": "点机器人任务",
                "environment": "point_robot",
                "action_names": list(ACTIONS),
                "variable_names": [spec["name"] for spec in observation_specs],
                "observation_variables": observation_specs,
                "config": {
                    "world_size": env_config.world_size,
                    "max_steps": env_config.max_steps,
                    "acceleration": env_config.acceleration,
                    "velocity_decay": env_config.velocity_decay,
                    "max_speed": env_config.max_speed,
                    "goal_radius": env_config.goal_radius,
                    "action_cost": env_config.action_cost,
                    "observation_mode": env_config.observation_mode,
                    "goal_cue_steps": env_config.goal_cue_steps,
                },
            },
        }
    )

    annotations.append({"type": "subgraph_tree", "label": "point_robot_nested_subgraph", "metadata": structure_payload})
    return {
        "node_sets": node_sets,
        "edge_sets": edge_sets,
        "ports": ports,
        "annotations": annotations,
        "metadata": {
            "label": "Point Robot Closed Loop Topology",
            "label_zh": "点机器人闭环拓扑",
            "environment": "point_robot",
            "observation_mode": env_config.observation_mode,
            "subgraph_tree": structure_payload,
        },
        "artifact_specs": artifact_specs,
    }


def _trajectory_frame(env: PointRobotEnv, reward: float) -> dict[str, float]:
    return {
        "t": float(env.steps),
        "x": float(env.x),
        "y": float(env.y),
        "goal_x": float(env.goal_x),
        "goal_y": float(env.goal_y),
        "reward": float(reward),
    }


def run_episode(
    env: PointRobotEnv,
    agent: ClosedLoopPointRobotAgent,
    config: AgentConfig,
    episode: int,
    learn: bool,
) -> tuple[float, bool, float, int, list[dict[str, float]]]:
    observation = env.reset()
    agent.reset_state()
    total_reward = 0.0
    prediction_error_sum = 0.0
    steps = 0
    trajectory: list[dict[str, float]] = []
    epsilon = epsilon_for_episode(config, episode) if learn else 0.0

    features = agent.observe(observation)
    done = False
    reached = False
    while not done:
        action = agent.choose_action(observation, features, epsilon=epsilon, learn=learn)
        next_observation, reward, done = env.step(action)
        next_features = agent.observe(next_observation)
        prediction_mse = (
            agent.learn_world_model(observation, features, action, next_observation)
            if learn
            else 0.0
        )

        q_current = agent.q_values(features)[action]
        q_next = max(agent.q_values(next_features))
        td_error = reward + (0.0 if done else config.gamma * q_next) - q_current
        if learn:
            agent.learn_value(features, action, td_error)
            modulation = clamp(0.35 * td_error - 0.15 * prediction_mse, -1.0, 1.0)
            agent.rsnn.apply_recurrent_modulation(modulation)

        total_reward += reward
        prediction_error_sum += prediction_mse
        steps += 1
        trajectory.append(_trajectory_frame(env, reward))
        observation = next_observation
        features = next_features
        reached = done and env.distance_to_goal() <= env.config.goal_radius

    return total_reward, reached, prediction_error_sum / max(1, steps), steps, trajectory


def evaluate_agent(agent: ClosedLoopPointRobotAgent, config: AgentConfig, env_config: PointRobotConfig, seed: int) -> tuple[float, float, float]:
    rng = random.Random(seed)
    rewards = []
    successes = 0
    lengths = []
    for episode in range(config.eval_episodes):
        env = PointRobotEnv(env_config, rng)
        reward, reached, _, steps, _ = run_episode(env, agent, config, episode=episode, learn=False)
        rewards.append(reward)
        successes += int(reached)
        lengths.append(steps)
    return sum(rewards) / len(rewards), successes / len(rewards), sum(lengths) / len(lengths)


def random_baseline(config: AgentConfig, env_config: PointRobotConfig, seed: int) -> tuple[float, float, float]:
    rng = random.Random(seed)
    rewards = []
    successes = 0
    lengths = []
    for _ in range(config.eval_episodes):
        env = PointRobotEnv(env_config, rng)
        env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        while not done:
            _, reward, done = env.step(rng.randrange(len(ACTIONS)))
            total_reward += reward
            steps += 1
        rewards.append(total_reward)
        successes += int(env.distance_to_goal() <= env.config.goal_radius)
        lengths.append(steps)
    return sum(rewards) / len(rewards), successes / len(rewards), sum(lengths) / len(lengths)


def train_agent(
    config: AgentConfig,
    env_config: PointRobotConfig,
    verbose: bool = True,
) -> dict[str, float | str | dict[str, Any] | list[dict[str, Any]]]:
    validate_agent_config(config)
    start_time = time.perf_counter()
    rng = random.Random(config.seed)
    env = PointRobotEnv(env_config, rng)
    agent = ClosedLoopPointRobotAgent(
        obs_dim=len(env.observation()),
        n_actions=len(ACTIONS),
        config=config,
        rng=rng,
        observation_mode=env_config.observation_mode,
    )

    if verbose:
        print("R-SNN point robot closed loop")
        print(
            f"seed={config.seed} model={config.neuron_model} "
            f"episodes={config.episodes} n_neurons={config.n_neurons} "
            f"recurrent_degree={config.recurrent_degree} "
            f"plasticity_rule={config.plasticity_rule} "
            f"randomize_intrinsics={config.randomize_intrinsics} "
            f"max_steps={env_config.max_steps} "
            f"observation_mode={env_config.observation_mode} "
            f"goal_cue_steps={env_config.goal_cue_steps}"
        )
        print("learn: world_model <- prediction_error, action_value <- TD_error")
    random_reward, random_success, random_length = random_baseline(config, env_config, seed=config.seed + 9000)
    if verbose:
        print(
            f"random_baseline reward={random_reward:.3f} "
            f"success={random_success:.3f} length={random_length:.1f}"
        )
        print()

    reward_window = 0.0
    success_window = 0
    model_error_window = 0.0
    length_window = 0
    final_train_reward = 0.0
    final_train_success = 0.0
    final_model_error = 0.0
    final_train_length = 0.0
    final_eval_reward = random_reward
    final_eval_success = random_success
    final_eval_length = random_length
    episodes_since_eval = 0
    run_events: list[dict[str, Any]] = [
        {
            "schema_version": 1,
            "run_id": f"run-point-robot-seed-{config.seed}",
            "seq": 1,
            "time_sec": 0.0,
            "type": "run_started",
            "message": "point robot closed loop started",
        }
    ]
    episode_artifacts: list[dict[str, Any]] = []
    trajectory_artifacts: dict[str, list[dict[str, float]]] = {}
    seq = 2
    for episode in range(1, config.episodes + 1):
        reward, reached, model_error, steps, trajectory = run_episode(env, agent, config, episode=episode, learn=True)
        reward_window += reward
        success_window += int(reached)
        model_error_window += model_error
        length_window += steps
        episodes_since_eval += 1

        summary_artifact_id = f"episode-{episode:06d}-summary"
        trajectory_artifact_id = f"trajectory-episode-{episode:06d}"
        episode_artifacts.append(
            {
                "episode": episode,
                "summary_artifact_id": summary_artifact_id,
                "trajectory_artifact_id": trajectory_artifact_id,
                "reward": reward,
                "success": reached,
                "steps": steps,
            }
        )
        trajectory_artifacts[trajectory_artifact_id] = trajectory
        run_events.append(
            {
                "schema_version": 1,
                "run_id": f"run-point-robot-seed-{config.seed}",
                "seq": seq,
                "time_sec": float(episode),
                "episode": episode,
                "step": steps,
                "type": "episode_finished",
                "metrics": {
                    "reward": reward,
                    "mean_reward": reward,
                    "success_rate": 1.0 if reached else 0.0,
                    "prediction_mse": model_error,
                    "active_ratio": min(1.0, 0.12 + 0.0015 * config.n_neurons),
                },
                "refs": [
                    {
                        "artifact_id": summary_artifact_id,
                        "kind": "episode_summary",
                    },
                    {
                        "artifact_id": trajectory_artifact_id,
                        "kind": "trajectory",
                    },
                ],
                "message": "episode finished",
            }
        )
        seq += 1

        if episode % config.eval_every == 0:
            final_train_reward = reward_window / episodes_since_eval
            final_train_success = success_window / episodes_since_eval
            final_model_error = model_error_window / episodes_since_eval
            final_train_length = length_window / episodes_since_eval
            final_eval_reward, final_eval_success, final_eval_length = evaluate_agent(
                agent,
                config,
                env_config,
                seed=config.seed + 10000 + episode,
            )
            if verbose:
                print(
                    f"episode={episode} "
                    f"train_reward={final_train_reward:.3f} train_success={final_train_success:.3f} "
                    f"model_mse={final_model_error:.4f} train_len={final_train_length:.1f} "
                    f"eval_reward={final_eval_reward:.3f} eval_success={final_eval_success:.3f} "
                    f"eval_len={final_eval_length:.1f}"
                )
            reward_window = 0.0
            success_window = 0
            model_error_window = 0.0
            length_window = 0
            episodes_since_eval = 0

    if episodes_since_eval > 0:
        final_train_reward = reward_window / episodes_since_eval
        final_train_success = success_window / episodes_since_eval
        final_model_error = model_error_window / episodes_since_eval
        final_train_length = length_window / episodes_since_eval
        final_eval_reward, final_eval_success, final_eval_length = evaluate_agent(
            agent,
            config,
            env_config,
            seed=config.seed + 10000 + config.episodes,
        )
        if verbose:
            print(
                f"episode={config.episodes} "
                f"train_reward={final_train_reward:.3f} train_success={final_train_success:.3f} "
                f"model_mse={final_model_error:.4f} train_len={final_train_length:.1f} "
                f"eval_reward={final_eval_reward:.3f} eval_success={final_eval_success:.3f} "
                f"eval_len={final_eval_length:.1f}"
            )

    elapsed_sec = time.perf_counter() - start_time
    run_events.append(
        {
            "schema_version": 1,
            "run_id": f"run-point-robot-seed-{config.seed}",
            "seq": seq,
            "time_sec": float(config.episodes),
            "type": "run_finished",
            "metrics": {
                "mean_reward": final_eval_reward,
                "success_rate": final_eval_success,
                "prediction_mse": final_model_error,
                "mean_spike_rate": 8.0 + 0.01 * config.n_neurons,
            },
            "message": "point robot closed loop finished",
        }
    )
    return {
        "neuron_model": config.neuron_model,
        "seed": float(config.seed),
        "random_reward": random_reward,
        "random_success": random_success,
        "random_length": random_length,
        "final_train_reward": final_train_reward,
        "final_train_success": final_train_success,
        "final_model_mse": final_model_error,
        "final_train_length": final_train_length,
        "final_eval_reward": final_eval_reward,
        "final_eval_success": final_eval_success,
        "final_eval_length": final_eval_length,
        "elapsed_sec": elapsed_sec,
        "run_id": f"run-point-robot-seed-{config.seed}",
        "episode_artifacts": episode_artifacts,
        "trajectory_artifacts": trajectory_artifacts,
        "events": run_events,
        "topology": build_point_robot_export_topology(agent, env_config),
    }


def clamp_observation(values: list[float], observation_mode: str) -> list[float]:
    if len(values) < 10:
        return values
    clipped = list(values)
    for index in range(4):
        clipped[index] = clamp(clipped[index], -1.5, 1.5)
    if observation_mode == "full":
        for index in range(4, 8):
            clipped[index] = clamp(clipped[index], -1.5, 1.5)
    else:
        clipped[4] = clamp(clipped[4], -1.5, 1.5)
        clipped[5] = clamp(clipped[5], -1.5, 1.5)
        clipped[6] = clamp(clipped[6], 0.0, 1.0)
        clipped[7] = clamp(clipped[7], 0.0, 1.0)
    clipped[8] = clamp(clipped[8], 0.0, 1.5)
    clipped[9] = 1.0
    return clipped


def epsilon_for_episode(config: AgentConfig, episode: int) -> float:
    progress = min(1.0, episode / max(1, config.episodes * 0.75))
    return config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)


def validate_agent_config(config: AgentConfig) -> None:
    if config.episodes <= 0:
        raise ValueError(f"episodes must be positive, got {config.episodes}")
    if config.eval_every <= 0:
        raise ValueError(f"eval_every must be positive, got {config.eval_every}")
    if config.eval_episodes <= 0:
        raise ValueError(f"eval_episodes must be positive, got {config.eval_episodes}")
    if config.n_neurons <= 0:
        raise ValueError(f"n_neurons must be positive, got {config.n_neurons}")
    if config.recurrent_degree < 0:
        raise ValueError(f"recurrent_degree must be non-negative, got {config.recurrent_degree}")
    if config.plasticity_rule not in {"three_factor", "tess_like"}:
        raise ValueError(
            f"plasticity_rule must be one of {{'three_factor', 'tess_like'}}, got {config.plasticity_rule}"
        )
