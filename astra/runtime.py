"""Local adaptations that stop failed evaluations without resampling a layout."""
import ast


def strict_eval_cli_source(source):
    """Prevent Isaac's preliminary parse from treating --device as --device_id."""
    tree = ast.parse(source)
    matches = [node for node in tree.body if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == 'parser' for target in node.targets)
               and isinstance(node.value, ast.Call)
               and isinstance(node.value.func, ast.Attribute)
               and isinstance(node.value.func.value, ast.Name)
               and node.value.func.value.id == 'argparse'
               and node.value.func.attr == 'ArgumentParser']
    if len(matches) != 1:
        raise ValueError('Unknown evaluation CLI; refusing ambiguous device arguments')
    lines = source.splitlines(keepends=True)
    lines.insert(matches[0].end_lineno, 'parser.allow_abbrev = False  # Keep --device distinct from --device_id.\n')
    return ''.join(lines)


def fail_closed_eval_source(source):
    tree = ast.parse(source)
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.Try)
               and any(isinstance(statement, ast.Expr)
                       and isinstance(statement.value, ast.Call)
                       and isinstance(statement.value.func, ast.Attribute)
                       and isinstance(statement.value.func.value, ast.Name)
                       and statement.value.func.value.id == 'env'
                       and statement.value.func.attr == 'run_eval'
                       for statement in node.body)]
    if len(matches) != 1 or not matches[0].handlers:
        raise ValueError('Unknown evaluation entrypoint; refusing unverified retry semantics')
    lines = source.splitlines(keepends=True)
    for handler in reversed(matches[0].handlers):
        # Insert before each handler body, including PhysX and unstable-scene paths.
        position = handler.body[0].lineno - 1
        indent = ' ' * handler.body[0].col_offset
        lines.insert(position, indent + 'raise  # Astra: preserve failed attempt; never resample automatically.\n')
    result = ''.join(lines)
    ast.parse(result)
    return result
