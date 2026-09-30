"""
Abstract Syntax Tree (AST) analyzer and codebase structural miner.
Extracts module metrics, cyclomatic complexity, function call graphs, and import hierarchies.
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

from project_marl.core.constants import EdgeType, WEIGHT_AST_CALL, WEIGHT_IMPORT
from project_marl.core.schemas import CodebaseModuleNode, DependencyEdge
from project_marl.core.logging import setup_logger

logger = setup_logger("ast_parser")


class ComplexityVisitor(ast.NodeVisitor):
    """
    Computes McCabe cyclomatic complexity of an AST.
    Complexity = 1 + number of branching decision points.
    """

    def __init__(self) -> None:
        self.complexity: int = 1

    def visit_If(self, node: ast.If) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self.complexity += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        # e.g. a and b and c adds (3 - 1) = 2 decision branches
        self.complexity += max(len(node.values) - 1, 1)
        self.generic_visit(node)

    def visit_Match(self, node: ast.AST) -> None:
        # Python 3.10+ match statement
        if hasattr(node, "cases"):
            self.complexity += len(getattr(node, "cases"))
        self.generic_visit(node)


class CodeASTAnalyzer:
    """
    Static analysis engine extracting code structure, complexity, and inter-module dependencies.
    """

    def __init__(self, root_dir: Optional[str] = None) -> None:
        self.root_dir = Path(root_dir).resolve() if root_dir else Path.cwd().resolve()
        self.modules: Dict[str, CodebaseModuleNode] = {}
        # Mapping from defined function name to list of module_ids defining it
        self.function_registry: Dict[str, List[str]] = {}

    def analyze_file(self, file_path: str) -> Optional[CodebaseModuleNode]:
        """
        Parses a single source file, computing AST complexity, definitions, and calls.
        """
        path = Path(file_path).resolve()
        if not path.is_file():
            logger.warning(f"File not found: {file_path}")
            return None

        try:
            rel_path = path.relative_to(self.root_dir).as_posix()
        except ValueError:
            rel_path = path.as_posix()

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return None

        lines_of_code = len([line for line in content.splitlines() if line.strip() and not line.strip().startswith("#")])

        try:
            tree = ast.parse(content, filename=str(path))
        except SyntaxError as e:
            logger.warning(f"Syntax error parsing {file_path}: {e}")
            # Fallback module node
            node = CodebaseModuleNode(
                module_id=rel_path,
                file_path=str(path),
                language=self._detect_language(path),
                lines_of_code=lines_of_code,
                cyclomatic_complexity=1.0,
            )
            node.compute_feature_vector()
            self.modules[rel_path] = node
            return node

        # Compute cyclomatic complexity
        visitor = ComplexityVisitor()
        visitor.visit(tree)
        complexity = float(visitor.complexity)

        # Extract functions, classes, imports, and function calls
        functions_defined: List[str] = []
        classes_defined: List[str] = []
        imports: List[str] = []
        functions_called: Set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions_defined.append(node.name)
                self.function_registry.setdefault(node.name, []).append(rel_path)
            elif isinstance(node, ast.ClassDef):
                classes_defined.append(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
            elif isinstance(node, ast.Call):
                call_name = self._get_call_name(node.func)
                if call_name:
                    functions_called.add(call_name)

        module_node = CodebaseModuleNode(
            module_id=rel_path,
            file_path=str(path),
            language=self._detect_language(path),
            lines_of_code=lines_of_code,
            cyclomatic_complexity=complexity,
            functions_count=len(functions_defined),
            classes_count=len(classes_defined),
            imports=sorted(list(set(imports))),
            functions_defined=sorted(functions_defined),
            functions_called=sorted(list(functions_called)),
        )
        module_node.compute_feature_vector()
        self.modules[rel_path] = module_node
        return module_node

    def analyze_directory(self, target_dir: Optional[str] = None, exclude_dirs: Optional[Set[str]] = None) -> Dict[str, CodebaseModuleNode]:
        """
        Recursively analyzes all source code files within target directory.
        """
        base_dir = Path(target_dir).resolve() if target_dir else self.root_dir
        if exclude_dirs is None:
            exclude_dirs = {".git", ".venv", "venv", "__pycache__", "node_modules", "dist", "build", ".worktrees"}

        logger.info(f"Scanning directory: {base_dir}")
        for root, dirs, files in os.walk(base_dir):
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith(".")]
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    self.analyze_file(full_path)

        logger.info(f"Analyzed {len(self.modules)} modules.")
        return self.modules

    def build_code_dependency_edges(self) -> List[DependencyEdge]:
        """
        Infers inter-module dependencies from function calls and import declarations.
        Returns a list of DependencyEdge objects.
        """
        edges: List[DependencyEdge] = []
        edge_set: Set[Tuple[str, str, EdgeType]] = set()

        for src_id, src_mod in self.modules.items():
            # 1. Check imports: does src_mod import any other module in self.modules?
            for imp in src_mod.imports:
                for target_id in self.modules.keys():
                    if src_id == target_id:
                        continue
                    # Match import path, e.g. 'project_marl.core' matching 'src/project_marl/core/schemas.py'
                    mod_dot_path = target_id.replace("/", ".").replace("\\", ".").removesuffix(".py")
                    if imp in mod_dot_path or mod_dot_path.endswith(imp):
                        key = (src_id, target_id, EdgeType.IMPLICIT_IMPORT)
                        if key not in edge_set:
                            edge_set.add(key)
                            edges.append(
                                DependencyEdge(
                                    source=src_id,
                                    target=target_id,
                                    edge_type=EdgeType.IMPLICIT_IMPORT,
                                    weight=WEIGHT_IMPORT,
                                    metadata={"imported_name": imp},
                                )
                            )

            # 2. Check function calls: does src_mod call functions defined in target_mod?
            for called_fn in src_mod.functions_called:
                # Simple function name lookup
                simple_fn = called_fn.split(".")[-1]
                target_modules = self.function_registry.get(simple_fn, [])
                for target_id in target_modules:
                    if target_id == src_id:
                        continue
                    key = (src_id, target_id, EdgeType.IMPLICIT_AST_CALL)
                    if key not in edge_set:
                        edge_set.add(key)
                        edges.append(
                            DependencyEdge(
                                source=src_id,
                                target=target_id,
                                edge_type=EdgeType.IMPLICIT_AST_CALL,
                                weight=WEIGHT_AST_CALL,
                                metadata={"call_name": called_fn},
                            )
                        )

        logger.info(f"Built {len(edges)} code dependency edges (AST calls & imports).")
        return edges

    def _get_call_name(self, node: ast.AST) -> Optional[str]:
        """Extracts human-readable function or method name from an ast.Call.func."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            val = self._get_call_name(node.value)
            return f"{val}.{node.attr}" if val else node.attr
        return None

    def _detect_language(self, path: Path) -> str:
        ext = path.suffix.lower()
        if ext == ".py":
            return "python"
        elif ext in (".ts", ".tsx"):
            return "typescript"
        elif ext in (".js", ".jsx"):
            return "javascript"
        elif ext == ".go":
            return "go"
        return "unknown"
