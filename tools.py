"""The two tools the agent can call. Neither uses eval()."""
import ast
import math
import operator

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.FloorDiv: operator.floordiv}
_UN = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FN = {"sqrt": math.sqrt, "cbrt": math.cbrt, "abs": abs, "round": round, "sin": math.sin,
       "cos": math.cos, "tan": math.tan, "asin": math.asin, "acos": math.acos, "atan": math.atan,
       "ln": math.log, "log": math.log10, "log2": math.log2, "exp": math.exp,
       "floor": math.floor, "ceil": math.ceil, "min": min, "max": max, "pow": math.pow,
       "radians": math.radians, "degrees": math.degrees}
_CONST = {"pi": math.pi, "e": math.e}


def calculator(expression: str) -> str:
    """Evaluate a maths expression safely by walking its syntax tree."""
    expr = str(expression).replace("^", "**").strip()
    if not expr or len(expr) > 200:
        raise ValueError("expression empty or too long")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError("could not parse the expression") from exc

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, (ast.Div, ast.Mod, ast.FloorDiv)) and right == 0:
                raise ValueError("division by zero")
            if isinstance(node.op, ast.Pow) and abs(right) > 1000:
                raise ValueError("exponent too large")
            return _BIN[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UN:
            return _UN[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Name) and node.id.lower() in _CONST:
            return _CONST[node.id.lower()]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id.lower() in _FN and not node.keywords:
            return _FN[node.func.id.lower()](*[ev(a) for a in node.args])
        raise ValueError("unsupported syntax in expression")

    result = ev(tree)
    if isinstance(result, complex) or not math.isfinite(result):
        raise ValueError("result is not a finite number")
    return format(float(result), ".10g")


UNITS = {
    "length": {"m": 1, "km": 1000, "cm": .01, "mm": .001, "mi": 1609.344, "yd": .9144, "ft": .3048, "in": .0254},
    "mass": {"kg": 1, "g": .001, "mg": 1e-6, "lb": .45359237, "oz": .028349523125, "t": 1000},
    "volume": {"l": 1, "ml": .001, "gal": 3.785411784, "qt": .946352946, "cup": .2365882365, "floz": .0295735295625},
    "time": {"s": 1, "ms": .001, "min": 60, "h": 3600, "day": 86400, "week": 604800},
    "speed": {"mps": 1, "kmh": 1 / 3.6, "mph": .44704, "knot": .514444},
    "area": {"m2": 1, "km2": 1e6, "cm2": 1e-4, "ha": 1e4, "acre": 4046.8564224, "ft2": .09290304},
    "data": {"bit": .125, "byte": 1, "kb": 1024, "mb": 1048576, "gb": 1073741824, "tb": 1099511627776},
}
ALIASES = {
    "meter": "m", "metre": "m", "kilometer": "km", "kilometre": "km", "centimeter": "cm", "millimeter": "mm",
    "mile": "mi", "yard": "yd", "foot": "ft", "feet": "ft", "inch": "in", "kilogram": "kg", "gram": "g",
    "milligram": "mg", "pound": "lb", "lbs": "lb", "ounce": "oz", "tonne": "t", "liter": "l", "litre": "l",
    "milliliter": "ml", "gallon": "gal", "quart": "qt", "second": "s", "sec": "s", "millisecond": "ms",
    "minute": "min", "hour": "h", "hr": "h", "kph": "kmh", "km/h": "kmh", "m/s": "mps", "mi/h": "mph",
    "knots": "knot", "sqm": "m2", "sqft": "ft2", "hectare": "ha", "b": "byte", "bytes": "byte", "bits": "bit",
    "kilobyte": "kb", "megabyte": "mb", "gigabyte": "gb", "terabyte": "tb",
    "celsius": "c", "fahrenheit": "f", "kelvin": "k",
}


def _canon(unit: str) -> str:
    x = str(unit).strip().lower().replace("°", "").replace(".", "").replace(" ", "")
    for cand in (x, x[:-1] if x.endswith("s") else x):
        c = ALIASES.get(cand, cand)
        if c in ("c", "f", "k") or any(c in t for t in UNITS.values()):
            return c
    raise ValueError(f"unknown unit '{unit}'")


def unit_converter(value, from_unit: str, to_unit: str) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("value must be a number") from exc
    a, b = _canon(from_unit), _canon(to_unit)
    if a in ("c", "f", "k") and b in ("c", "f", "k"):
        celsius = v if a == "c" else (v - 32) * 5 / 9 if a == "f" else v - 273.15
        out = celsius if b == "c" else celsius * 9 / 5 + 32 if b == "f" else celsius + 273.15
    else:
        table = next((t for t in UNITS.values() if a in t and b in t), None)
        if table is None:
            raise ValueError(f"cannot convert '{from_unit}' to '{to_unit}' (different categories)")
        out = v * table[a] / table[b]
    return f"{v:g} {a} = {float(format(out, '.8g')):g} {b}"


TOOLS = {
    "calculator": (calculator, 'args {"expression": str}. Operators + - * / ** % ( ). Functions sqrt, cbrt, abs, round, floor, ceil, sin, cos, tan, asin, acos, atan (radians; use radians(x) for degrees), ln, log (base 10), log2, exp, min, max, pow. Constants pi, e.'),
    "unit_converter": (unit_converter, 'args {"value": number, "from_unit": str, "to_unit": str}. Length, mass, volume, time, speed, area, data (1 KB=1024 B), temperature.'),
}


def run_tool(name: str, args: dict):
    """Return (ok, text). Never raises."""
    try:
        return True, TOOLS[name][0](**(args or {}))
    except Exception as exc:  # tool errors go back to the model so it can fix its arguments
        return False, f"{type(exc).__name__}: {exc}"
