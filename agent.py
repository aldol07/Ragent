"""Two-agent pipeline:

1. Researcher - a tool-calling loop (search_documents / list_documents / calculator)
   that streams a cited answer.
2. Verifier   - checks every claim in the draft against the retrieved passages and,
   if something is unsupported, rewrites the answer using only the evidence.
"""
import json
import re
import time

from config import HISTORY_TURNS, MAX_AGENT_STEPS, TOP_K_RESULTS
from tools import TOOL_SCHEMAS, calculate

RESEARCHER_PROMPT = """You are Ragent, an assistant that answers questions about the user's uploaded documents.

Rules:
- For anything about the documents, call search_documents first. Split multi-part or comparison questions into several focused searches. Rewrite follow-up questions into self-contained queries.
- Write search queries about the content you need, not file names; use the `source` argument to restrict a search to one file.
- Use calculator for any arithmetic; never do math in your head.
- Answer ONLY from the passages returned by the tools. Cite every factual sentence with the passage numbers in plain square brackets, e.g. [1] or [2][4]. Calculator results need no citation.
- If the passages do not contain the answer, say so plainly instead of guessing.
- Be concise: lead with the direct answer, then supporting detail as short bullets if useful. Use Markdown.
Available documents: {documents}"""

VERIFIER_PROMPT = """You are a strict fact-checker. Compare the ANSWER with the numbered EVIDENCE passages.
A claim is supported only if the evidence states or directly implies it. Calculator results count as evidence.
Return JSON only: {{"supported": true|false, "unsupported_claims": ["..."]}}

EVIDENCE:
{evidence}

QUESTION: {question}

ANSWER:
{answer}"""

REVISER_PROMPT = """Rewrite the ANSWER so that it contains only claims supported by the EVIDENCE.
Remove or correct these unsupported claims: {claims}
Keep the [n] citations, keep the Markdown style, and if little is left, say what the documents do and do not cover.

EVIDENCE:
{evidence}

QUESTION: {question}

ANSWER:
{answer}

Return only the revised answer."""

NO_DOCS_NOTE = "none uploaded yet"

# Some models cite as 【3】 or 【3†L1-L4】 (OpenAI browsing style); normalise to [3].
_ODD_CITATION = re.compile(r"【\s*(\d+)[^】]*】")
_NON_NUMERIC_CITATION = re.compile(r"\s*【[^】\d]*】")


_LEAKED_TOOL_CALL = re.compile(r"<tool_call>.*?(</tool_call>|$)", re.S)

FINAL_STEP_NUDGE = (
    "You have reached the search limit. Do not call any more tools. "
    "Write the final answer now using only the passages above, with [n] citations."
)


def clean_answer(text):
    """Drop tool calls a model wrote as plain text, and normalise citations."""
    return normalize_citations(_LEAKED_TOOL_CALL.sub("", text)).strip()


def normalize_citations(text):
    text = _ODD_CITATION.sub(lambda m: f"[{m.group(1)}]", text)
    return _NON_NUMERIC_CITATION.sub("", text)


class Agent:
    def __init__(self, llm, retriever=None, verify=True, top_k=TOP_K_RESULTS):
        self.llm = llm
        self.retriever = retriever
        self.verify = verify
        self.top_k = top_k

    # ------------------------------------------------------------------ tools
    def _run_tool(self, name, args, state):
        if name == "search_documents":
            if not self.retriever:
                return "No documents are loaded.", "no documents"
            query = (args.get("query") or "").strip()
            if not query:
                return "Empty query.", "empty query"
            hits = self.retriever.search(query, k=self.top_k, source=args.get("source"))
            lines = []
            for hit in hits:
                chunk = hit["chunk"]
                key = (chunk.source, chunk.chunk_id)
                if key not in state["index"]:
                    state["index"][key] = len(state["sources"]) + 1
                    state["sources"].append({**hit, "n": state["index"][key]})
                n = state["index"][key]
                lines.append(f"[{n}] ({chunk.citation})\n{chunk.text}")
            text = "\n\n".join(lines) or "No matching passages."
            return text, f"{len(hits)} passages"

        if name == "list_documents":
            if not self.retriever:
                return "No documents are loaded.", "no documents"
            return json.dumps(self.retriever.sources), f"{len(self.retriever.sources)} documents"

        if name == "calculator":
            expr = args.get("expression", "")
            try:
                result = calculate(expr)
                state["calculations"].append(f"{expr} = {result}")
                return str(result), f"= {result}"
            except Exception as e:
                return f"Error: {e}", f"error: {e}"

        return f"Unknown tool {name}", "unknown tool"

    # ------------------------------------------------------------ main loop
    def run(self, question, history=None):
        """Generator of events:

        {"type": "tool", "name", "args", "result"}   a tool call finished
        {"type": "token", "text"}                     streamed answer text
        {"type": "revised", "answer"}                 verifier replaced the draft
        {"type": "final", "answer", "sources", "verification", "stats"}
        """
        started = time.time()
        self.llm.reset_usage()
        documents = (
            ", ".join(s["source"] for s in self.retriever.sources) if self.retriever else NO_DOCS_NOTE
        )
        messages = [{"role": "system", "content": RESEARCHER_PROMPT.format(documents=documents)}]
        for turn in (history or [])[-2 * HISTORY_TURNS:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": question})

        state = {"sources": [], "index": {}, "calculations": []}
        tool_calls_made = []
        answer = ""
        llm_steps = 0

        for step in range(MAX_AGENT_STEPS):
            llm_steps += 1
            final_step = step == MAX_AGENT_STEPS - 1
            if final_step:
                messages.append({"role": "user", "content": FINAL_STEP_NUDGE})
            message = None
            stream = self.llm.stream(messages, tools=TOOL_SCHEMAS, tool_choice="none" if final_step else None)
            for kind, payload in stream:
                if kind == "text":
                    yield {"type": "token", "text": payload}
                else:
                    message = payload

            if final_step:
                message.pop("tool_calls", None)  # ignore any call the model makes anyway
            messages.append(message)
            if not message.get("tool_calls"):
                answer = clean_answer(message["content"])
                break

            if message["content"]:
                # Text emitted before a tool call is "thinking aloud", not the answer.
                yield {"type": "reset"}

            for call in message["tool_calls"]:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                output, summary = self._run_tool(name, args, state)
                tool_calls_made.append({"name": name, "args": args, "result": summary})
                yield {"type": "tool", "name": name, "args": args, "result": summary}
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})

        if not answer:
            answer = "I couldn't finish answering within the step limit. Please try a narrower question."

        verification = None
        if self.verify and state["sources"]:
            verification = self._verify(question, answer, state)
            if verification and not verification.get("supported", True):
                revised = self._revise(question, answer, state, verification.get("unsupported_claims", []))
                if revised:
                    answer = clean_answer(revised) or answer
                    verification["revised"] = True
                    yield {"type": "revised", "answer": answer}

        cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
        sources = [s for s in state["sources"] if s["n"] in cited] or state["sources"]

        yield {
            "type": "final",
            "answer": answer,
            "sources": sources,
            "verification": verification,
            "tool_calls": tool_calls_made,
            "stats": {
                "latency_s": round(time.time() - started, 2),
                "model": getattr(self.llm, "last_model", None),
                "steps": llm_steps,
                **self.llm.usage,
            },
        }

    def ask(self, question, history=None):
        """Non-streaming convenience wrapper; returns the final event."""
        final = None
        for event in self.run(question, history):
            if event["type"] == "final":
                final = event
        return final

    # ------------------------------------------------------------- verifier
    @staticmethod
    def _evidence(state):
        parts = [f"[{s['n']}] ({s['chunk'].citation}) {s['chunk'].text}" for s in state["sources"]]
        parts += [f"Calculator: {c}" for c in state["calculations"]]
        return "\n\n".join(parts)

    def _verify(self, question, answer, state, attempts=2):
        prompt = VERIFIER_PROMPT.format(evidence=self._evidence(state), question=question, answer=answer)
        result = None
        for _ in range(attempts):  # free reasoning models occasionally return an empty message
            try:
                result = self.llm.complete_json(
                    [{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=2000,  # reasoning models think before emitting the JSON
                )
            except Exception as e:
                print(f"[verifier] failed: {e}")
            if isinstance(result, dict):
                break
        if not isinstance(result, dict):
            print("[verifier] could not parse a verdict")
            return None
        return {
            "supported": bool(result.get("supported", True)),
            "unsupported_claims": [str(c) for c in result.get("unsupported_claims") or []],
            "revised": False,
        }

    def _revise(self, question, answer, state, claims):
        try:
            return self.llm.complete(
                [{"role": "user", "content": REVISER_PROMPT.format(
                    claims="; ".join(claims) or "(unspecified)",
                    evidence=self._evidence(state), question=question, answer=answer)}],
                temperature=0.0,
                max_tokens=2000,
            )
        except Exception as e:
            print(f"[reviser] failed: {e}")
            return None
