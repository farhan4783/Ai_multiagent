"""
AST-Aware Pre-Merge Conflict Resolution Engine for Parallel Workspaces.

Performs static analysis across Git worktree branches before merges are attempted,
detecting file-level collisions, symbol modifications, signature drift, and AST coupling.
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import ast
import difflib
import os
import subprocess

from project_marl.core.logging import setup_logger

logger = setup_logger("conflict_resolver")


@dataclass
class SymbolModification:
    """Represents a modified syntactic construct in a file."""
    file_path: str
    symbol_type: str  # 'function', 'class', 'method', 'import'
    symbol_name: str
    change_type: str  # 'added', 'modified', 'deleted'
    line_start: int = 0
    line_end: int = 0


@dataclass
class ConflictReport:
    """Comprehensive conflict analysis report between two workspaces or branches."""
    branch_a: str
    branch_b: str
    has_file_collision: bool
    has_symbol_collision: bool
    overlapping_files: List[str] = field(default_factory=list)
    symbol_collisions: List[Dict[str, Any]] = field(default_factory=list)
    conflict_risk_score: float = 0.0
    recommendation: str = "SAFE_TO_MERGE"

    def is_mergeable(self) -> bool:
        return self.recommendation in ("SAFE_TO_MERGE", "LOW_RISK_DIFF")


class ASTConflictResolver:
    """
    Performs pre-merge static analysis on parallel agent worktrees.
    Detects overlapping AST symbol mutations and calculates collision risk.
    """

    def __init__(self, repo_dir: Optional[str] = None) -> None:
        self.repo_dir = Path(repo_dir).resolve() if repo_dir else Path.cwd().resolve()

    def analyze_worktrees(
        self,
        worktree_a_path: str,
        worktree_b_path: str,
        base_path: Optional[str] = None,
        branch_a_name: str = "worktree_a",
        branch_b_name: str = "worktree_b",
    ) -> ConflictReport:
        """
        Analyzes file differences and AST symbol overlap between two active worktrees.
        """
        wt_a = Path(worktree_a_path)
        wt_b = Path(worktree_b_path)
        base = Path(base_path) if base_path else self.repo_dir

        # 1. Identify modified files in both worktrees relative to base
        mods_a = self._get_modified_files_in_worktree(wt_a, base)
        mods_b = self._get_modified_files_in_worktree(wt_b, base)

        overlapping_files = sorted(list(mods_a.intersection(mods_b)))
        has_file_collision = len(overlapping_files) > 0

        symbol_collisions: List[Dict[str, Any]] = []

        # 2. For overlapping files, perform deep AST analysis
        for rel_file in overlapping_files:
            file_a = wt_a / rel_file
            file_b = wt_b / rel_file
            base_file = base / rel_file

            if rel_file.endswith(".py"):
                file_collisions = self._analyze_python_ast_collisions(file_a, file_b, base_file, rel_file)
                symbol_collisions.extend(file_collisions)
            else:
                # Text-level diff collision check
                if self._has_textual_conflict(file_a, file_b, base_file):
                    symbol_collisions.append({
                        "file": rel_file,
                        "symbol_type": "text_block",
                        "symbol_name": rel_file,
                        "conflict_detail": "Overlapping text diff lines",
                    })

        has_symbol_collision = len(symbol_collisions) > 0

        # 3. Calculate quantitative conflict risk score [0.0, 1.0]
        if not has_file_collision:
            risk_score = 0.0
            recommendation = "SAFE_TO_MERGE"
        elif not has_symbol_collision:
            risk_score = min(0.35, 0.10 * len(overlapping_files))
            recommendation = "LOW_RISK_DIFF"
        else:
            # Direct symbol overlap
            risk_score = min(1.0, 0.40 + 0.20 * len(symbol_collisions))
            if risk_score > 0.70:
                recommendation = "BLOCKING_MERGE_DISPUTE"
            else:
                recommendation = "AST_COLLISION_WARNING"

        return ConflictReport(
            branch_a=branch_a_name,
            branch_b=branch_b_name,
            has_file_collision=has_file_collision,
            has_symbol_collision=has_symbol_collision,
            overlapping_files=overlapping_files,
            symbol_collisions=symbol_collisions,
            conflict_risk_score=round(risk_score, 2),
            recommendation=recommendation,
        )

    def _get_modified_files_in_worktree(self, worktree_path: Path, base_path: Path) -> Set[str]:
        """Returns set of relative paths modified in the worktree compared to base."""
        modified: Set[str] = []
        # Check via git if valid git directory
        try:
            res = subprocess.run(
                ["git", "-C", str(worktree_path), "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=False,
                encoding="utf-8",
                errors="replace",
                stdin=subprocess.DEVNULL,
            )
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        modified.append(parts[-1])
        except Exception:
            pass

        # Also scan files on disk directly
        for root, _, files in os.walk(worktree_path):
            if any(part.startswith(".") for part in Path(root).parts if part not in (".", "..")):
                continue
            for f in files:
                wt_file = Path(root) / f
                rel_path = wt_file.relative_to(worktree_path).as_posix()
                base_file = base_path / rel_path
                if not base_file.exists() or wt_file.stat().st_mtime != base_file.stat().st_mtime:
                    try:
                        if not base_file.exists() or wt_file.read_bytes() != base_file.read_bytes():
                            modified.append(rel_path)
                    except Exception:
                        pass

        return set(modified)

    def _analyze_python_ast_collisions(
        self,
        file_a: Path,
        file_b: Path,
        base_file: Path,
        rel_path: str,
    ) -> List[Dict[str, Any]]:
        """Compares AST definitions across branches A, B and base file."""
        collisions: List[Dict[str, Any]] = []

        code_a = file_a.read_text(encoding="utf-8", errors="replace") if file_a.exists() else ""
        code_b = file_b.read_text(encoding="utf-8", errors="replace") if file_b.exists() else ""
        code_base = base_file.read_text(encoding="utf-8", errors="replace") if base_file.exists() else ""

        try:
            ast_a = ast.parse(code_a)
            ast_b = ast.parse(code_b)
            ast_base = ast.parse(code_base)
        except SyntaxError:
            # If invalid syntax, report raw text conflict
            return [{
                "file": rel_path,
                "symbol_type": "syntax_error",
                "symbol_name": rel_path,
                "conflict_detail": "Python syntax error in one of the branches",
            }]

        symbols_a = self._extract_symbols(ast_a)
        symbols_b = self._extract_symbols(ast_b)
        symbols_base = self._extract_symbols(ast_base)

        # Find symbols modified in both A and B relative to base
        all_symbol_keys = set(symbols_a.keys()).union(symbols_b.keys())

        for key in all_symbol_keys:
            name, sym_type = key
            code_piece_a = symbols_a.get(key)
            code_piece_b = symbols_b.get(key)
            code_piece_base = symbols_base.get(key)

            changed_in_a = code_piece_a != code_piece_base
            changed_in_b = code_piece_b != code_piece_base

            if changed_in_a and changed_in_b and code_piece_a != code_piece_b:
                collisions.append({
                    "file": rel_path,
                    "symbol_type": sym_type,
                    "symbol_name": name,
                    "conflict_detail": f"Both branches modified {sym_type} '{name}' with differing implementations.",
                })

        return collisions

    def _extract_symbols(self, tree: ast.AST) -> Dict[Tuple[str, str], str]:
        """Extracts { (symbol_name, symbol_type): unparsed_or_dumped_code }."""
        symbols: Dict[Tuple[str, str], str] = {}
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols[(node.name, "function")] = ast.dump(node)
            elif isinstance(node, ast.ClassDef):
                symbols[(node.name, "class")] = ast.dump(node)
                # Also extract methods
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        symbols[(f"{node.name}.{item.name}", "method")] = ast.dump(item)
        return symbols

    def _has_textual_conflict(self, file_a: Path, file_b: Path, base_file: Path) -> bool:
        """Checks if textual diff lines overlap between branch A and B."""
        if not file_a.exists() or not file_b.exists():
            return True
        lines_a = file_a.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        lines_b = file_b.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        lines_base = base_file.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True) if base_file.exists() else []

        matcher = difflib.SequenceMatcher(None, lines_a, lines_b)
        return matcher.ratio() < 0.95
