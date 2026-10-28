"""Pruebas del 03: causalidad, contrato contable, algoritmo y aislamiento temporal.

Ejecutar desde la raíz: python -m unittest discover -s tests -v
"""
import unittest
import numpy as np

from src.env import SimpleTradingEnv
from src.experiment import run_episode
from src.markets import make_prices, EXTRA_MARKETS
from src.extras import (causal_features, prepare, fit_encoder, FeatureTradingEnv,
                        make_agent, run_extra_episode, real_data, real_indices,
                        evaluation_series)


class ExtrasTests(unittest.TestCase):
    def setUp(self):
        self.prices = make_prices("regimenes", 160, 7)
        self.prepared = prepare(self.prices)
        self.cfg = dict(algoritmo="qlearning", seed=0, alpha=.001, gamma=.9,
                        epsilon_decay=.99, epsilon_min=.01)

    def test_future_prices_do_not_change_past_features(self):
        changed = self.prices.copy()
        changed[80:] *= np.linspace(1.5, 2, len(changed)-80)
        np.testing.assert_array_equal(causal_features(self.prices)[:80], causal_features(changed)[:80])
        np.testing.assert_array_equal(causal_features(self.prices[:80]), causal_features(self.prices)[:80])

    def test_subclass_preserves_rewards_and_accounting(self):
        original = SimpleTradingEnv(self.prices)
        enriched = FeatureTradingEnv(self.prepared)
        original.reset(); enriched.reset()
        for action in np.random.default_rng(3).integers(3, size=len(self.prices)-6):
            a = original.step(int(action)); b = enriched.step(int(action))
            self.assertEqual(a[1:], b[1:])
            self.assertAlmostEqual(float(a[0][0]), float(b[0][0]), places=7)
            self.assertEqual(a[0][1], b[0][-1])
            self.assertTrue(enriched.observation_space.contains(b[0]))

    def test_base_encoding_preserves_shared_qlearning_and_sarsa(self):
        encoder = fit_encoder([self.prepared])
        for algorithm in ("qlearning", "sarsa"):
            with self.subTest(algorithm=algorithm):
                cfg = dict(self.cfg, algoritmo=algorithm)
                a, b = make_agent(cfg, encoder), make_agent(cfg, encoder)
                shared = run_episode(SimpleTradingEnv(self.prices), a, learn=True)
                extended = run_extra_episode(self.prepared, b, encoder, learn=True, greedy=False)
                for x, y in zip(shared, extended):
                    np.testing.assert_allclose(x, y)
                np.testing.assert_allclose(a.Q, b.Q)

    def test_evaluation_does_not_update_q(self):
        encoder = fit_encoder([self.prepared], "enriquecido")
        agent = make_agent(self.cfg, encoder)
        run_extra_episode(self.prepared, agent, encoder, learn=True, greedy=False)
        before = agent.Q.copy()
        run_extra_episode(self.prepared, agent, encoder, learn=False)
        np.testing.assert_array_equal(before, agent.Q)

    def test_mixed_radix_encoding_and_duplicate_quantiles(self):
        encoder = fit_encoder([self.prepared], "enriquecido")
        self.assertEqual(encoder.n_states, 180)
        codes = encoder.codes(self.prepared.features)
        self.assertTrue(np.all(codes % 2 == 0))
        self.assertTrue(np.all((codes >= 0) & (codes + 1 < encoder.n_states)))
        # Una serie plana no debe producir bins repetidos ni índices fuera de Q.
        flat = prepare(np.ones(80) * 100)
        flat_encoder = fit_encoder([flat], "enriquecido")
        self.assertTrue(all(np.all(np.diff(e) > 0) for e in flat_encoder.edges))
        self.assertLessEqual(flat_encoder.codes(flat.features).max()+1, flat_encoder.n_states-1)

    def test_real_split_boundaries_and_warmup(self):
        df, _ = real_data()  # También comprueba SHA-256 del fichero local.
        train, val, test = [real_indices(s) for s in ("train", "val", "test")]
        self.assertLess(train[-1], val[0]); self.assertLess(val[-1], test[0])
        self.assertEqual(len(set(train) & set(val)), 0)
        self.assertEqual(len(set(train) & set(test)), 0)
        p = evaluation_series("sp500_real", "test")[0]
        self.assertEqual(p.dates[5], df.date.iloc[test[0]].to_datetime64())
        self.assertEqual(p.dates[-1], df.date.iloc[test[-1]].to_datetime64())

    def test_extra_markets_have_positive_reproducible_prices(self):
        for market in EXTRA_MARKETS:
            a = make_prices(market, 300, 123)
            b = make_prices(market, 300, 123)
            np.testing.assert_array_equal(a, b)
            self.assertTrue(np.all(np.isfinite(a) & (a > 0)))
            self.assertEqual(len(a), 300)


if __name__ == "__main__":
    unittest.main()
