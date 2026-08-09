# InjecAgent evaluation

Evaluation snapshot: 2026-08-09.

We evaluated Llama-3.1-8B and Llama-3.3-70B on the complete released
[InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) test set: 1,054
indirect prompt-injection cases per condition, covering direct harm and
two-stage data stealing. Results use InjecAgent commit
`f19c9f2c79a41046eb13c03c51a24c567a8ffa07`.

Attack success rate (ASR) is the fraction of all 1,054 cases in which the
released evaluator finds the target attacker tool action. Lower is better.
Benign utility (BU) is semantic success on paired clean tasks, judged blind by
`gpt-4o-mini`; higher is better. The base and enhanced attack settings are
reported separately.

## Llama-3.1-8B

| Defense | Base ASR | Enhanced ASR | Clean BU |
|---|---:|---:|---:|
| No defense | 68.79% | 81.12% | 83.11% |
| **ODILE** | **4.08%** | **0.00%** | 70.49% |
| Meta-SecAlign | 11.67% | 9.39% | 74.00% |
| ReasAlign | 38.80% | 12.05% | **85.96%** |
| MELON | 5.98% | 2.75% | 74.29% |
| Prompt Guard 2 | 69.64% | 1.71% | 80.46% |
| Repeat user prompt | 46.11% | 55.79% | 76.38% |
| Spotlighting | 68.12% | 79.51% | 78.84% |
| Instruction hierarchy | 65.65% | 71.16% | 78.75% |
| DeBERTa PI detector v2 | 15.28% | 1.23% | 47.15% |
| Sanitizer | 43.17% | 34.35% | 71.16% |

## Llama-3.3-70B

| Defense | Base ASR | Enhanced ASR | Clean BU |
|---|---:|---:|---:|
| No defense | 74.29% | 90.23% | 97.44% |
| **ODILE** | 3.51% | **0.00%** | **97.63%** |
| Meta-SecAlign | 8.44% | 18.79% | 87.67% |
| MELON | **2.75%** | 1.23% | 95.83% |
| Prompt Guard 2 | 73.62% | 1.14% | **97.63%** |
| Repeat user prompt | 17.27% | 44.31% | 94.02% |
| Spotlighting | 71.92% | 87.95% | 93.17% |
| Instruction hierarchy | 70.30% | 87.57% | 92.60% |
| DeBERTa PI detector v2 | 23.72% | 1.04% | 55.50% |
| Sanitizer | 15.28% | 11.01% | 87.95% |

ReasAlign is omitted at 70B because its released defense checkpoint is 8B.
Runtime defenses are applied to the corresponding unmodified Llama base model.

## ODILE versus Meta-SecAlign

| Size / attack | ODILE ASR | Meta-SecAlign ASR | Difference |
|---|---:|---:|---:|
| 8B base | **4.08%** | 11.67% | ODILE lower by 7.59 points |
| 8B enhanced | **0.00%** | 9.39% | ODILE lower by 9.39 points |
| 70B base | **3.51%** | 8.44% | ODILE lower by 4.93 points |
| 70B enhanced | **0.00%** | 18.79% | ODILE lower by 18.79 points |

Paired exact McNemar tests favor ODILE on 8B base (`p=2.69e-11`), 8B
enhanced (`p=3.16e-30`), 70B base (`p=9.44e-8`), and 70B enhanced
(`p=4.98e-60`). The 8B clean-utility difference between ODILE and
Meta-SecAlign is not conventionally significant (`p=0.0545`); ODILE has higher
clean utility at 70B (`p=7.27e-25`).

## Protocol and interpretation

- ODILE and the native base control use an explicit function-result boundary.
  Meta-SecAlign uses its released `input` role and tokenizer/chat template;
  giving it this expected serialization is essential. A matched base control
  under that serialization has 84.54% clean utility at 8B and 87.29% at 70B.
- ODILE is trained to circuit-break when it recognizes an injected trajectory,
  so attacked output collapse is intended behavior rather than ordinary task
  completion. The 8B ODILE run reached the generation cap in 915/1,054 base
  and 1,054/1,054 enhanced cases; the 70B run did so in 998/1,054 base and
  1,054/1,054 enhanced cases. ASR should therefore be read together with the
  clean-utility column, not as a standalone capability score.
- Raw-text audits found no parser-inflated ODILE result. The exact relevant
  attacker-action scan found 51 actions versus 43 official successes at 8B
  base, zero versus zero at 8B enhanced, 42 versus 37 at 70B base, and zero
  versus zero at 70B enhanced. The released parser is slightly conservative.
- Prompt Guard 2's low enhanced ASR partly reflects its fixed warning response.
  A small number of continuations still select the expected attacker tool,
  including calls that merely echo a warning but satisfy InjecAgent's released
  tool-name success rule.
- AttriGuard is excluded: its interrupted partial traces did not satisfy the
  1,054-case coverage gate.
