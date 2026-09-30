"""
Unit tests for CodeASTAnalyzer and ComplexityVisitor in project_marl.graph.ast_parser.
"""

import tempfile
import os
from pathlib import Path
from project_marl.graph.ast_parser import CodeASTAnalyzer, ComplexityVisitor
from project_marl.core.constants import EdgeType


def test_complexity_visitor() -> None:
    code = """
def complex_function(x, y, z):
    if x > 0:
        for i in range(y):
            if i % 2 == 0 and z > 10:
                print("even and high")
            elif i % 2 != 0:
                print("odd")
    while x < 100:
        x += 10
    try:
        assert x > 0
    except AssertionError:
        pass
    return [k for k in range(x) if k > 5]
"""
    import ast
    tree = ast.parse(code)
    visitor = ComplexityVisitor()
    visitor.visit(tree)
    # 1 (base) + 1 (if) + 1 (for) + 1 (if) + 1 (and) + 1 (elif) + 1 (while) + 1 (assert) + 1 (except) + 1 (comprehension) + 1 (comprehension if) = 11
    assert visitor.complexity >= 10


def test_ast_analyzer_on_temp_directory() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        mod_a = Path(tmpdir) / "module_a.py"
        mod_b = Path(tmpdir) / "module_b.py"

        mod_a.write_text(
            """
def compute_delta(a, b):
    if a > b:
        return a - b
    return b - a
""",
            encoding="utf-8",
        )

        mod_b.write_text(
            """
import module_a

def orchestrate():
    val = compute_delta(10, 5)
    return val
""",
            encoding="utf-8",
        )

        analyzer = CodeASTAnalyzer(tmpdir)
        analyzer.analyze_directory(tmpdir)

        assert len(analyzer.modules) == 2
        node_a = analyzer.modules.get("module_a.py")
        assert node_a is not None
        assert "compute_delta" in node_a.functions_defined
        assert node_a.cyclomatic_complexity == 2.0  # base 1 + if 1

        node_b = analyzer.modules.get("module_b.py")
        assert node_b is not None
        assert "module_a" in node_b.imports
        assert "compute_delta" in node_b.functions_called

        # Test edge building
        edges = analyzer.build_code_dependency_edges()
        # Should have import edge and/or AST call edge from module_b -> module_a
        src_targets = [(e.source, e.target, e.edge_type) for e in edges]
        assert any(s == "module_b.py" and t == "module_a.py" for s, t, _ in src_targets)
