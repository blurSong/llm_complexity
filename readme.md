# A simple calculator to evaluate the math/dram complexity of llm inference.

## Usage

```python
from llm_complexity import auto_model, calc_inference_complexity

model = auto_model("deepseek-ai/DeepSeek-V3-0324")
calc_inference_complexity(model, 1024, 128, 1, "a16w4", verbose=False)
```

## Result
**Simple results**
```bash
╭──────────────────┬───────────────┬─────────────┬─────────┬──────────┬──────────┬─────────────────────┬─────────────┬──────────────────╮
│ Model            │ Phase         │ Precision   │ Batch   │ Prompt   │ Output   │ Required DRAM GBs   │ Math OPs    │ Total IO Bytes   │
├──────────────────┼───────────────┼─────────────┼─────────┼──────────┼──────────┼─────────────────────┼─────────────┼──────────────────┤
│ DeepSeek-V3-0324 │ prefill       │ a16w4       │ 1       │ 1024     │ 512      │ 172.16              │ 8.02657e+13 │ 1.84282e+11      │
├──────────────────┼───────────────┼─────────────┼─────────┼──────────┼──────────┼─────────────────────┼─────────────┼──────────────────┤
│ DeepSeek-V3-0324 │ decode (once) │ a16w4       │ 1       │ 1024     │ 512      │ 17.5885             │ 7.96728e+10 │ 1.40222e+10      │
╰──────────────────┴───────────────┴─────────────┴─────────┴──────────┴──────────┴─────────────────────┴─────────────┴──────────────────╯
```

**[Verbose result](/results/)**
