#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Tuple, Callable, List

import toh_mdp as tm


def value_iteration(
        mdp: tm.TohMdp, v_table: tm.VTable
) -> Tuple[tm.VTable, tm.QTable, float]:
    """Computes one step of value iteration.

    Hint 1: Since the terminal state will always have value 0 since
    initialization, you only need to update values for nonterminal states.

    Hint 2: It might be easier to first populate the Q-value table.

    Args:
        mdp: the MDP definition.
        v_table: Value table from the previous iteration.

    Returns:
        new_v_table: tm.VTable
            New value table after one step of value iteration.
        q_table: tm.QTable
            New Q-value table after one step of value iteration.
        max_delta: float
            Maximum absolute value difference for all value updates, i.e.,
            max_s |V_k(s) - V_k+1(s)|.
    """
    new_v_table: tm.VTable = v_table.copy()
    q_table: tm.QTable = {}
    # noinspection PyUnusedLocal
    max_delta = 0.0
    # *** BEGIN OF YOUR CODE ***
    # bellman update: U(s) = max_a sum_s' T(s, a, s') * (R(s, a, s') + gamma * U(s'))
    for state in mdp.nonterminal_states:
        best_q_value = float("-inf")

        for action in mdp.actions:
            q_value = 0.0
            for next_state in mdp.all_states:
                transition_probability = mdp.transition(state, action, next_state)
                if transition_probability == 0.0:  # skip invalid next states
                    continue

                reward = mdp.reward(state, action, next_state)
                q_value += transition_probability * (
                    reward + mdp.config.gamma * v_table[next_state]
                )

            q_table[(state, action)] = q_value
            if q_value > best_q_value:
                best_q_value = q_value

        new_v_table[state] = best_q_value
        delta = abs(new_v_table[state] - v_table[state])
        max_delta = max(max_delta, delta)
    # ***  END OF YOUR CODE  ***
    return new_v_table, q_table, max_delta


def extract_policy(
        mdp: tm.TohMdp, q_table: tm.QTable
) -> tm.Policy:
    """Extract policy mapping from Q-value table.

    Remember that no action is available from the terminal state, so the
    extracted policy only needs to have all the nonterminal states (can be
    accessed by mdp.nonterminal_states) as keys.

    Args:
        mdp: the MDP definition.
        q_table: Q-Value table to extract policy from.

    Returns:
        policy: tm.Policy
            A Policy maps nonterminal states to actions.
    """
    # *** BEGIN OF YOUR CODE ***
    policy = {}
    for state in mdp.nonterminal_states:
        policy[state] = max(mdp.actions, key=lambda action: q_table[(state, action)])
    return policy


def q_update(
        mdp: tm.TohMdp, q_table: tm.QTable,
        transition: Tuple[tm.TohState, tm.TohAction, float, tm.TohState],
        alpha: float) -> None:
    """Perform a Q-update based on a (S, A, R, S') transition.

    Update the relevant entries in the given q_update based on the given
    (S, A, R, S') transition and alpha value.

    Args:
        mdp: the MDP definition.
        q_table: the Q-Value table to be updated.
        transition: A (S, A, R, S') tuple representing the agent transition.
        alpha: alpha value (i.e., learning rate) for the Q-Value update.
    """
    state, action, reward, next_state = transition
    # *** BEGIN OF YOUR CODE ***
    old_q_value = q_table[(state, action)]
    next_state_best_q = max(q_table[(next_state, next_action)] for next_action in mdp.actions) if next_state != mdp.terminal else 0
    # formula from lecture slides: Q(S, A) <- (1 - alpha) * Q(S, A) + alpha * (R + gamma * max_a' Q(S', a'))
    q_table[(state, action)] = (1.0 - alpha) * old_q_value + alpha * (reward + mdp.config.gamma * next_state_best_q)


def extract_v_table(mdp: tm.TohMdp, q_table: tm.QTable) -> tm.VTable:
    """Extract the value table from the Q-Value table.

    Args:
        mdp: the MDP definition.
        q_table: the Q-Value table to extract values from.

    Returns:
        v_table: tm.VTable
            The extracted value table.
    """
    # *** BEGIN OF YOUR CODE ***
    v_table = {}
    for state in mdp.nonterminal_states:
        v_table[state] = max(q_table[(state, action)] for action in mdp.actions)
    v_table[mdp.terminal] = 0.0
    return v_table


def choose_next_action(
        mdp: tm.TohMdp, state: tm.TohState, epsilon: float, q_table: tm.QTable,
        epsilon_greedy: Callable[[List[tm.TohAction], float], tm.TohAction]
) -> tm.TohAction:
    """Use the epsilon greedy function to pick the next action.

    You can assume that the passed in state is neither the terminal state nor
    any goal state.

    You can think of the epsilon greedy function passed in having the following
    definition:

    def epsilon_greedy(best_actions, epsilon):
        # selects one of the best actions with probability 1-epsilon,
        # selects a random action with probability epsilon
        ...

    See the concrete definition in QLearningSolver.epsilon_greedy.

    Args:
        mdp: the MDP definition.
        state: the current MDP state.
        epsilon: epsilon value in epsilon greedy.
        q_table: the current Q-value table.
        epsilon_greedy: a function that performs the epsilon

    Returns:
        action: tm.TohAction
            The chosen action.
    """
    # *** BEGIN OF YOUR CODE ***
    best_q_value = max(q_table[(state, action)] for action in mdp.actions)
    best_actions = [action for action in mdp.actions if q_table[(state, action)] == best_q_value]
    return epsilon_greedy(best_actions, epsilon)


def custom_epsilon(n_step: int) -> float:
    """Calculates the epsilon value for the nth Q learning step.

    Define a function for epsilon based on `n_step`.

    Args:
        n_step: the nth step for which the epsilon value will be used.

    Returns:
        epsilon: float
            epsilon value when choosing the nth step.
    """
    # *** BEGIN OF YOUR CODE ***


def custom_alpha(n_step: int) -> float:
    """Calculates the alpha value for the nth Q learning step.

    Define a function for alpha based on `n_step`.

    Args:
        n_step: the nth update for which the alpha value will be used.

    Returns:
        alpha: float
            alpha value when performing the nth Q update.
    """
    # *** BEGIN OF YOUR CODE ***
