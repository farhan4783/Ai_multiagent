"""
Git commit co-churn and historical coupling miner.
Analyzes version control commit histories to detect evolutionary dependencies and file churn.
"""

import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

from project_marl.core.constants import EdgeType, WEIGHT_CO_CHURN, MIN_CO_CHURN_THRESHOLD
from project_marl.core.schemas import DependencyEdge
from project_marl.core.logging import setup_logger

logger = setup_logger("git_miner")


class GitChurnMiner:
    """
    Extracts commit histories, analyzes module co-change frequencies,
    and infers evolutionary coupling between codebase files.
    """

    def __init__(self, repo_dir: Optional[str] = None) -> None:
        self.repo_dir = Path(repo_dir).resolve() if repo_dir else Path.cwd().resolve()
        # file_path -> total commit count touching it
        self.file_commit_counts: Dict[str, int] = defaultdict(int)
        # (file_a, file_b) -> number of commits modifying both
        self.co_change_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        # file_path -> total lines changed (rough churn volume)
        self.file_churn_volume: Dict[str, int] = defaultdict(int)

    def is_git_repo(self) -> bool:
        """Verifies if the target directory is inside a valid git repository."""
        git_dir = self.repo_dir / ".git"
        return git_dir.exists()

    def mine_commit_history(self, max_commits: int = 500) -> List[Dict[str, any]]:
        """
        Executes git log to extract commit metadata and modified file sets.
        """
        if not self.is_git_repo():
            logger.warning(f"Directory {self.repo_dir} is not a git repository.")
            return []

        cmd = [
            "git",
            "-C",
            str(self.repo_dir),
            "log",
            f"-n{max_commits}",
            "--name-only",
            "--format=COMMIT_META:%H|%an|%at",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                encoding="utf-8",
                errors="replace",
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"Failed to execute git log: {e}")
            return []

        commits: List[Dict[str, any]] = []
        current_commit: Optional[Dict[str, any]] = None
        current_files: Set[str] = set()

        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue

            if line.startswith("COMMIT_META:"):
                if current_commit is not None and current_files:
                    current_commit["files"] = sorted(list(current_files))
                    commits.append(current_commit)
                    self._record_commit(current_files)

                parts = line.removeprefix("COMMIT_META:").split("|")
                current_commit = {
                    "hash": parts[0] if len(parts) > 0 else "",
                    "author": parts[1] if len(parts) > 1 else "",
                    "timestamp": float(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0.0,
                    "files": [],
                }
                current_files = set()
            else:
                # Standardize path separators to forward slash
                clean_path = line.replace("\\", "/").strip()
                if clean_path:
                    current_files.add(clean_path)

        if current_commit is not None and current_files:
            current_commit["files"] = sorted(list(current_files))
            commits.append(current_commit)
            self._record_commit(current_files)

        logger.info(f"Mined {len(commits)} commits touching {len(self.file_commit_counts)} distinct files.")
        return commits

    def _record_commit(self, files: Set[str]) -> None:
        """Updates commit counts and pairwise co-change tallies."""
        file_list = sorted(list(files))
        for f in file_list:
            self.file_commit_counts[f] += 1

        n = len(file_list)
        # Avoid exploding pairwise combinations for massive bulk commits (e.g. initial commit of 1000 files)
        if n > 50:
            return

        for i in range(n):
            for j in range(i + 1, n):
                pair = (file_list[i], file_list[j])
                self.co_change_counts[pair] += 1

    def build_co_churn_edges(
        self,
        min_threshold: float = MIN_CO_CHURN_THRESHOLD,
        filter_modules: Optional[Set[str]] = None,
    ) -> List[DependencyEdge]:
        """
        Computes Jaccard normalized co-churn coupling between files and constructs DependencyEdge instances.
        Jaccard(A, B) = CoCount(A, B) / (Count(A) + Count(B) - CoCount(A, B))
        """
        edges: List[DependencyEdge] = []

        for (file_a, file_b), co_count in self.co_change_counts.items():
            if filter_modules:
                if file_a not in filter_modules and file_b not in filter_modules:
                    continue

            count_a = self.file_commit_counts.get(file_a, 1)
            count_b = self.file_commit_counts.get(file_b, 1)

            union_count = count_a + count_b - co_count
            jaccard = co_count / union_count if union_count > 0 else 0.0

            if jaccard >= min_threshold or co_count >= 2:
                weight = float(jaccard * WEIGHT_CO_CHURN)
                # Bidirectional co-churn relationship
                edge_ab = DependencyEdge(
                    source=file_a,
                    target=file_b,
                    edge_type=EdgeType.IMPLICIT_CO_CHURN,
                    weight=weight,
                    metadata={"co_churn_count": co_count, "jaccard": jaccard},
                )
                edge_ba = DependencyEdge(
                    source=file_b,
                    target=file_a,
                    edge_type=EdgeType.IMPLICIT_CO_CHURN,
                    weight=weight,
                    metadata={"co_churn_count": co_count, "jaccard": jaccard},
                )
                edges.append(edge_ab)
                edges.append(edge_ba)

        logger.info(f"Built {len(edges)} co-churn evolutionary coupling edges.")
        return edges

    def get_module_churn_score(self, module_id: str) -> float:
        """Returns relative churn frequency score for a given module."""
        # Check direct or suffix match
        count = self.file_commit_counts.get(module_id, 0)
        if count == 0:
            for k, v in self.file_commit_counts.items():
                if k.endswith(module_id) or module_id.endswith(k):
                    count = max(count, v)
        # Log-scaled score
        import math
        return float(math.log1p(count))
