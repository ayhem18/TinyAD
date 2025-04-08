
from tinyad.parsing.parser import build_parentheses_tree

if __name__ == "__main__":
    expression = "((x + y) + (2 * x ^ 2 + 3 * y))"
    res = build_parentheses_tree(expression)
    print(res)

    expression = "(((2x) + (3y)) + ((4x) + (5y * (z + w))))"
    res = build_parentheses_tree(expression)
    print(res)