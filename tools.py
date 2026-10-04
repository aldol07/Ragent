"""Tools the agent can call, plus their JSON schemas for function calling."""
import ast
import math
import operator

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARYOPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "exp": math.exp,
    "abs": abs, "round": round, "min": min, "max": max,
}
_CONSTS = {"pi": math.pi, "e": math.e}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent too large.")
        return _BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCS
        and not node.keywords
    ):
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError("Unsupported expression. Use numbers, + - * / // % **, and sqrt/log/exp/abs/round/min/max.")


def calculate(expression):
    """Safely evaluate an arithmetic expression (no eval())."""
    expr = expression.replace("×", "*").replace("÷", "/").replace("^", "**").replace(",", "")
    if len(expr) > 200:
        raise ValueError("Expression too long.")
    result = _eval(ast.parse(expr, mode="eval"))
    if isinstance(result, float):
        result = round(result, 10)
    return result


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the uploaded documents and return the most relevant passages with "
                "numbered citations. Call this before answering anything about the documents. "
                "Call it several times with focused queries for multi-part questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A focused, self-contained search query."},
                    "source": {
                        "type": "string",
                        "description": "Optional exact file name to restrict the search to one document.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "List the uploaded documents with their page and chunk counts.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate an arithmetic expression exactly. Use for any math instead of computing mentally.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "e.g. (25 + 33) * 2 / 7"}},
                "required": ["expression"],
            },
        },
    },
]
