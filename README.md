## Evaluation

You should have `full_predictions.pt` file in the `benchmark` directory.

To run evaluation script, execute the following command:

```bash
python3 evaluate.py --k 20
```

The output metrics for each test fold will be saved in `benchmark/metrics` directory.
