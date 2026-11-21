# xrl-battery

Explainable reinforcement learning for home battery management: a PPO controller trained on the electricity bill, compared with a rule-based controller and explained with LIME.

- `notebooks/main_results.ipynb` - main results (training, test profit, behaviour, LIME, reward cases, regimes)
- `notebooks/side_experiments.ipynb` - ablations, other RL algorithms, training curves
- `notebooks/models/` - trained PPO models used by the main notebook

```
pip install -r requirements.txt
cd notebooks && jupyter notebook
```

Set `RETRAIN = True` in the main notebook to train the models from scratch. `data/Annual_load_PV_with_price.csv` is stored with Git LFS.
