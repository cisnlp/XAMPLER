# XAMPLER

Code for 'XAMPLER: Learning to Retrieve Cross-Lingual In-Context Examples'.

Pipeline:

1. Prepare sib200: refer to https://github.com/dadelani/sib-200.

2. Conduct 1-shot in-context learning: `icl/get_signal.py`

3. Train retriever: `train_retriever.py`

4. Get retrieved results: `internal_retriever.py`

5. Conduct in-context learning: `icl/eval.py`

