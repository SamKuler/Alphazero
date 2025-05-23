from .node import MCTSNode, INF
from .config import MCTSConfig
from env.base_env import BaseGame

import numpy as np


class UCTMCTSConfig(MCTSConfig):
    def __init__(
            self,
            n_rollout: int = 1,
            enable_heuristic: bool = False,
            heuristic_fn=None,
            max_rollout_depth: int = 10,
            *args, **kwargs
    ):
        MCTSConfig.__init__(self, *args, **kwargs)
        self.n_rollout = n_rollout
        self.enable_heuristic = enable_heuristic
        self.heuristic_fn = heuristic_fn
        self.max_rollout_depth = max_rollout_depth


class UCTMCTS:
    def __init__(self, init_env: BaseGame, config: UCTMCTSConfig, root: MCTSNode = None):
        self.config = config
        self.root = root
        if root is None:
            self.init_tree(init_env)
        self.root.cut_parent()

    def init_tree(self, init_env: BaseGame):
        # initialize the tree with the current state
        # fork the environment to avoid side effects
        env = init_env.fork()
        self.root = MCTSNode(
            action=None, env=env, reward=0,
        )

    def get_subtree(self, action: int):
        # return a subtree with root as the child of the current root
        # the subtree represents the state after taking action
        if self.root.has_child(action):
            new_root = self.root.get_child(action)
            return UCTMCTS(new_root.env, self.config, new_root)
        else:
            return None

    def uct_action_select(self, node: MCTSNode) -> int:
        # select the best action based on UCB when expanding the tree

        ########################
        # TODO: your code here #
        ########################
        n_total = np.sum(node.child_N_visit)
        valid_action_mask = np.where(node.action_mask > 0)[0]

        best_action = -1
        best_ucb1 = -np.inf

        # UCB1(s) = U(s)/N(s) + C*sqrt(log2(N)/N(s))

        for action in valid_action_mask:
            ns = node.child_N_visit[action]
            us = node.child_V_total[action]
            if ns == 0:
                ucb1 = np.inf
            else:
                ucb1 = (us / ns) + self.config.C * np.sqrt(np.log(n_total) / ns)
            if ucb1 > best_ucb1:
                best_ucb1 = ucb1
                best_action = action
        return best_action
        ########################

    def backup(self, node: MCTSNode, value: float) -> None:
        # backup the value of the leaf node to the root
        # update N_visit and V_total of each node in the path

        ########################
        # TODO: your code here #
        ########################
        cur = node
        while cur.parent is not None:
            parent = cur.parent
            parent.child_N_visit[cur.action] += 1
            parent.child_V_total[cur.action] += value
            value *= -1
            cur = parent
        ########################

    def rollout(self, node: MCTSNode) -> float:
        # simulate the game until the end
        # return the reward of the game
        # NOTE: the reward should be converted to the perspective of the current player!

        ########################
        # TODO: your code here #
        ########################
        env = node.env.fork()
        done = env.ended
        reward = 0
        # simulation
        if not self.config.enable_heuristic:
            while not done:
                valid_action_mask = np.where(env.action_mask > 0)[0]
                if len(valid_action_mask) == 0:
                    break
                action = np.random.choice(valid_action_mask)
                _, reward, done = env.step(action)
            if node.env.current_player != env.current_player:
                reward = -reward
            return reward
        else :
            depth = 0
            while not done and depth < self.config.max_rollout_depth:
                valid_action_mask = np.where(env.action_mask > 0)[0]
                if len(valid_action_mask) == 0:
                    break
                action = np.random.choice(valid_action_mask)
                _, reward, done = env.step(action)
                depth += 1
            if not done:
                heuristics_value = self.config.heuristic_fn(env)
                if env.current_player == node.env.current_player:
                    heuristics_value = -heuristics_value
                return heuristics_value
            if node.env.current_player != env.current_player:
                reward = -reward
            return reward
        ########################

    def pick_leaf(self) -> MCTSNode:
        # select the leaf node to expand
        # the leaf node is the node that has not been expanded
        # create and return a new node if game is not ended

        ########################
        # TODO: your code here #
        ########################
        cur = self.root
        while not cur.done:
            # valid_action_mask = np.where(cur.action_mask > 0)[0]
            # unvisited = [
            #     action for action in valid_action_mask if cur.child_N_visit[action] == 0
            # ]
            # if len(unvisited) > 0:
            #     # expand
            #     action = np.random.choice(unvisited)
            #     return cur.add_child(action)
            # # select
            # action = self.uct_action_select(cur)
            # cur = cur.get_child(action)

            # -------------------#
            action = self.uct_action_select(cur)
            if cur.has_child(action):
                cur = cur.get_child(action)
            else:
                return cur.add_child(action)

        return cur
        ########################

    def get_policy(self, node: MCTSNode = None) -> np.ndarray:
        # return the policy of the tree(root) after the search
        # the policy comes from the visit count of each action

        ########################
        # TODO: your code here #
        ########################
        if node is None:
            node = self.root

        normalize_factor = np.sum(node.child_N_visit)
        if normalize_factor > 0:
            return node.child_N_visit / normalize_factor
        else:
            # print("Default policy")
            policy = np.zeros(node.n_action)
            valid_action_mask = np.where(node.action_mask > 0)[0]
            policy[valid_action_mask] = 1 / len(valid_action_mask)
            return policy
        ########################

    def search(self):
        # search the tree for n_search times
        # eachtime, pick a leaf node, rollout the game (if game is not ended)
        #   for n_rollout times, and backup the value.
        # return the policy of the tree after the search
        for _ in range(self.config.n_search):
            leaf = self.pick_leaf()
            value = 0.0
            if leaf.done:
                ########################
                # TODO: your code here #
                ########################
                value = float(leaf.reward)
                ########################
            else:
                ########################
                # TODO: your code here #
                ########################
                for _ in range(self.config.n_rollout):
                    value += self.rollout(leaf)
                value /= self.config.n_rollout
                # print(value)
                ########################
            self.backup(leaf, value)

        return self.get_policy(self.root)