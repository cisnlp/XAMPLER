# XAMPLER: Learning to Retrieve Cross-Lingual In-Context Examples

[![arXiv](https://img.shields.io/badge/arXiv-2405.05116-b31b1b.svg)](https://arxiv.org/abs/2405.05116)

## Usage

1. Prepare sib200: refer to https://github.com/dadelani/sib-200.

2. Conduct 1-shot in-context learning: `icl/get_signal.py`

3. Train retriever: `train_retriever.py`

4. Get retrieved results: `internal_retriever.py`

5. Conduct in-context learning: `icl/eval.py`

## Citation

```
@misc{lin2024xampler,
    title={XAMPLER: Learning to Retrieve Cross-Lingual In-Context Examples},
    author={Peiqin Lin and André F. T. Martins and Hinrich Schütze},
    year={2024},
    eprint={2405.05116},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

