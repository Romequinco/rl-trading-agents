"""Pruebas del anexo del 03: equivalencias con el 03, comisión, nuevos algoritmos y placebo.

Ejecutar desde la raíz: python -m unittest discover -s tests -v
"""
import unittest
import numpy as np

from src.env import SimpleTradingEnv
from src.markets import make_prices
from src.extras import prepare, fit_encoder, make_agent as make_agent_03, run_extra_episode, real_indices
from src.extras_anexo import (CostTradingEnv, ExpectedSarsaAgent, DoubleQLearningAgent,
                              fit_subset_encoder, make_agent, run_episode_anexo, shuffled_real,
                              real_data)


class AnexoTests(unittest.TestCase):
    def setUp(self):
        self.prices = make_prices("reversion_media", 160, 7)
        self.prepared = prepare(self.prices)
        self.cfg = dict(algoritmo="qlearning", seed=0, alpha=.001, gamma=.9,
                        epsilon_decay=.99, epsilon_min=.01)

    def test_cost_env_reproduces_workshop_at_default_cost(self):
        original, costed = SimpleTradingEnv(self.prices), CostTradingEnv(self.prepared)
        original.reset(); costed.reset()
        for action in np.random.default_rng(5).integers(3, size=len(self.prices)-6):
            a, b = original.step(int(action)), costed.step(int(action))
            self.assertEqual(a[1:], b[1:])

    def test_zero_cost_buy_hold_matches_price_ratio(self):
        env = CostTradingEnv(self.prepared, 0.0)
        env.reset()
        done = False
        while not done:
            *_, done, _, _ = env.step(2)
        self.assertAlmostEqual(env.balance / 1000, self.prices[-1] / self.prices[5], places=9)

    def test_subset_encoder_matches_03_and_same_trajectory(self):
        for mode in ("base", "enriquecido"):
            with self.subTest(mode=mode):
                old, new = fit_encoder([self.prepared], mode), fit_subset_encoder([self.prepared], mode)
                np.testing.assert_array_equal(old.codes(self.prepared.features), new.codes(self.prepared.features))
                a, b = make_agent_03(self.cfg, old), make_agent(self.cfg, new)
                x = run_extra_episode(self.prepared, a, old, learn=True, greedy=False)
                y = run_episode_anexo(self.prepared, b, new, learn=True, greedy=False)
                for u, v in zip(x, y):
                    np.testing.assert_allclose(u, v)
                np.testing.assert_allclose(a.Q, b.Q)
        self.assertEqual(fit_subset_encoder([self.prepared], "ma30").n_states, 60)
        self.assertEqual(fit_subset_encoder([self.prepared], "vol").n_states, 60)

    def test_log_reward_sums_to_log_final_return(self):
        encoder = fit_subset_encoder([self.prepared], "base")
        total, balances, *_ = run_episode_anexo(self.prepared, make_agent(self.cfg, encoder), encoder,
                                                reward="log", learn=True, greedy=False)
        self.assertAlmostEqual(total, np.log(balances[-1] / balances[0]), places=9)

    def test_expected_sarsa_target(self):
        agent = ExpectedSarsaAgent(edges=[0.0], alpha=1.0, gamma=0.5, epsilon=0.3)
        agent.Q[1] = [0.0, 1.0, 4.0]
        agent.update(0, 2, 0.1, 1, None, False)
        expected = 0.1 + 0.5 * (0.1 * 0 + 0.1 * 1 + 0.8 * 4)
        self.assertAlmostEqual(agent.Q[0, 2], expected)
        agent.update(0, 1, 0.2, 1, None, True)
        self.assertAlmostEqual(agent.Q[0, 1], 0.2)

    def test_double_q_updates_one_table_with_the_other(self):
        agent = DoubleQLearningAgent(edges=[0.0], alpha=1.0, gamma=1.0, seed=0)
        agent.QA[1] = [0.0, 0.0, 1.0]; agent.QB[1] = [0.0, 5.0, 2.0]
        agent.update(0, 0, 0.0, 1, None, False)
        # La tabla elegida toma su argmax en S' y lo valora con la otra: 2 (A elige, B valora) o 0 (B elige, A valora).
        self.assertIn((agent.QA[0, 0], agent.QB[0, 0]), [(2.0, 0.0), (0.0, 0.0)])
        np.testing.assert_allclose(agent.Q, agent.QA + agent.QB)

    def test_shuffled_real_keeps_split_boundaries_and_destroys_order(self):
        _, full = real_data()
        fake = shuffled_real(0)
        for split in ("train", "val", "test"):
            idx = real_indices(split)
            self.assertAlmostEqual(fake.prices[idx[0]], full.prices[idx[0]], places=6)
        self.assertAlmostEqual(fake.prices[-1], full.prices[-1], places=6)
        self.assertFalse(np.allclose(fake.prices, full.prices))
        test = real_indices("test")
        r_real = np.sort(np.diff(np.log(full.prices[test[0]:])))
        r_fake = np.sort(np.diff(np.log(fake.prices[test[0]:])))
        np.testing.assert_allclose(r_real, r_fake, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
