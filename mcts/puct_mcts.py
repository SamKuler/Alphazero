from .node import MCTSNode, INF
from .config import MCTSConfig
from env.base_env import BaseGame

from model.linear_model_trainer import NumpyLinearModelTrainer
import numpy as np


class PUCTMCTS:
    def __init__(self, init_env:BaseGame, model: NumpyLinearModelTrainer, config: MCTSConfig, root:MCTSNode=None):
        self.model = model
        self.config = config
        self.root = root
        if root is None:
            self.init_tree(init_env)
        self.root.cut_parent()
    
    def init_tree(self, init_env:BaseGame):
        env = init_env.fork()
        obs = env.observation
        self.root = MCTSNode(
            action=None, env=env, reward=0
        )
        # compute and save predicted policy
        child_prior, _ = self.model.predict(env.compute_canonical_form_obs(obs, env.current_player))
        self.root.set_prior(child_prior)
    
    def get_subtree(self, action:int):
        # return a subtree with root as the child of the current root
        # the subtree represents the state after taking action
        if self.root.has_child(action):
            new_root = self.root.get_child(action)
            return PUCTMCTS(new_root.env, self.model, self.config, new_root)
        else:
            return None
    
    def puct_action_select(self, node:MCTSNode):
        # select the best action based on PUCB when expanding the tree
        
        ########################
        # TODO: your code here #
        ########################
        n_total = np.sum(node.child_N_visit)
        valid_action_mask = np.where(node.action_mask > 0)[0]

        best_action = -1
        best_pucb = -np.inf

        # PUCB(s) = V(c_i)/N(c_i) + C*P(s,c_i)*sqrt(sum(N(c_i)))/(1+N(c_i))
        for action in valid_action_mask:
            vs = node.child_V_total[action]
            ns = node.child_N_visit[action]
            ps = node.child_priors[action]

            if ns == 0:
                pucb = np.inf
            else:
                pucb  = vs/ns + self.config.C*ps*np.sqrt(n_total)/(1+ns)

            if pucb > best_pucb:
                best_pucb = pucb
                best_action = action

        return best_action
        ########################

    def backup(self, node:MCTSNode, value):
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
    
    def pick_leaf(self):
        # select the leaf node to expand
        # the leaf node is the node that has not been expanded
        # create and return a new node if game is not ended
        
        ########################
        # TODO: your code here #
        ########################
        # if self.config.with_noise and self.root.child_priors.sum() > 0\
        #     and np.allclose(self.root.child_N_visit,0):
        #     noise = np.random.dirichlet([self.config.dir_alpha]*self.root.n_action)
        #     self.root.child_priors = (1-self.config.dir_epsilon)*self.root.child_priors + self.config.dir_epsilon*noise

        cur = self.root
        while not cur.done:
            action = self.puct_action_select(cur)
            if cur.has_child(action):
                cur = cur.get_child(action)
            else:
                return cur.add_child(action)
        return cur
        ########################
    
    def get_policy(self, node:MCTSNode = None):
        # return the policy of the tree(root) after the search
        # the policy conmes from the visit count of each action 
        
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
                # NOTE: you should compute the policy and value 
                #       using the value&policy model!
                canonical_obs = leaf.env.get_canonical_form_obs()
                child_prior, value_pred = self.model.predict(canonical_obs)
                leaf.set_prior(child_prior)
                value = -value_pred[0]  # Extract scalar value, multiply -1 as view from root
                ########################
            self.backup(leaf, value)
            
        return self.get_policy(self.root)