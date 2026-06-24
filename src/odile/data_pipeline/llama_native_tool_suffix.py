"""Shared helper for the Llama native tool-instruction suffix.

Imported by both:
  - scripts/eval_lorra_multiturn.py   (eval-time prompt construction)
  - scripts/data_pipeline/generate_llama_completions.py
                                       (training-trace generation)

so the two pipelines produce byte-identical tool-suffix strings.

CoT NOTE: The instruction block below asks the model to write a short
natural-language rationale *before* the JSON tool call. This mirrors what the
PromptingLLM (XML) pipeline did with <function-thoughts>...</function-thoughts>
tags. Without CoT, the Llama-native bad/anchor completions start with
*identical* JSON tokens (`{"type": "function", "name": "send_money", …`), so
the LoRRA 15-token loss window captures no token-level content signal — only
prefix-driven hidden-state drift. With CoT, the first ~15 tokens contain the
reasoning text, which diverges naturally between bad and anchor (e.g.
"I need to transfer money to US133..." vs. "I need to pay the bill to
UK12..."), restoring the token-level discriminative signal.

The JSON parser in eval (`LlamaNativeLLM._extract_tool_payloads`) scans for
the first valid JSON object in the completion — the preceding reasoning text
is harmless to it.
"""

import os
from typing import Sequence

from agentdojo.functions_runtime import Function


_COT_INSTRUCTIONS = (
    "You have access to the following tools. For every response that needs "
    "a tool call, first state your reasoning in one or two short sentences "
    "of plain text (what the user is asking, which tool is appropriate, and "
    "why), then emit the tool call as a JSON object with \"name\" and "
    "\"parameters\" fields on its own line.\n"
    "\n"
    "Example:\n"
    "I need to check the user's account balance before paying the bill.\n"
    '{"name": "get_balance", "parameters": {}}'
)

# Plain instruction block used when CoT is disabled. Matches the style the
# original (pre-CoT) training-data generator used, so eval-time prompts can be
# made byte-identical to what the adapter saw during training.
_PLAIN_INSTRUCTIONS = (
    "You have access to the following tools. To call a tool, respond with "
    "a JSON object with \"name\" and \"parameters\" fields."
)


def build_llama_native_tool_suffix(tools: Sequence[Function]) -> str:
    """Build tool schema text in Llama's native tool-instruction style.

    Output layout (appended to the system message content):

        \\n\\n
        Environment: ipython
        Tools: <comma-separated names>

        # Tool Instructions

        <CoT instruction: reason first, then emit JSON; with example>

        ## <tool_name>

        <description>

        Parameters:
          - <pname> (<ptype>): <pdesc> (required|optional)
          ...
    """
    if not tools:
        return ""
    schemas = []
    for tool in tools:
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters.model_json_schema(),
                },
            }
        )

    # CoT toggle (default on; override via env var for non-CoT runs):
    #   LLAMA_NATIVE_COT=0  → plain "respond with JSON" block (matches the
    #                         original training-data generator — use this
    #                         for apples-to-apples eval on adapters trained
    #                         without CoT traces)
    #   LLAMA_NATIVE_COT=1  → CoT block (default; current training/eval)
    cot_mode = os.environ.get("LLAMA_NATIVE_COT", "1").strip().lower()
    instructions_block = _PLAIN_INSTRUCTIONS if cot_mode in ("0", "false", "no", "off") else _COT_INSTRUCTIONS

    lines = [
        "",
        "",
        "Environment: ipython",
        "Tools: " + ", ".join(t["function"]["name"] for t in schemas),
        "",
        "# Tool Instructions",
        "",
        instructions_block,
        "",
    ]

    for schema in schemas:
        fn = schema["function"]
        lines.append(f"## {fn['name']}")
        lines.append("")
        lines.append(fn["description"])
        lines.append("")
        lines.append("Parameters:")
        params = fn["parameters"]
        props = params.get("properties", {})
        required = set(params.get("required", []))
        for pname, pinfo in props.items():
            ptype = pinfo.get("type", "any")
            pdesc = pinfo.get("description", "")
            req = " (required)" if pname in required else " (optional)"
            lines.append(f"  - {pname} ({ptype}): {pdesc}{req}")
        lines.append("")

    return "\n".join(lines)
