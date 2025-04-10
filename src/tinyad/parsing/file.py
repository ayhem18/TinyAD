
from tinyad.parsing.parser import build_parentheses_tree, evaluate_expression


def f1():
    expression = "((x + y) + (2 * x ^ 2 + 3 * y))"
    res = build_parentheses_tree(expression)
    print(res)

    expression = "(((2x) + (3y)) + ((4x) + (5y * (z + w))))"
    res = build_parentheses_tree(expression)
    print(res)

    expression = "((2x) + (3y)) + ((4x) + (5y * (z + w)))"
    res = build_parentheses_tree(expression)
    print(res)


def f2():
    expression = "x y ^ 4 + x z + x * y ^ 2"
    var_name_tracker = {}
    res = evaluate_expression(expression, {}, var_name_tracker)
    print(res.name)


if __name__ == "__main__":
    f2()