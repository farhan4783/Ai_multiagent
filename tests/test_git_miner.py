"""
Unit tests for GitChurnMiner in project_marl.graph.git_miner.
"""

from project_marl.graph.git_miner import GitChurnMiner
from project_marl.core.constants import EdgeType


def test_git_churn_miner_on_current_repo() -> None:
    miner = GitChurnMiner(".")
    assert miner.is_git_repo() is True

    commits = miner.mine_commit_history(max_commits=10)
    assert len(commits) > 0

    edges = miner.build_co_churn_edges(min_threshold=0.0)
    # Check that any generated edges have correct type and weights
    for edge in edges:
        assert edge.edge_type == EdgeType.IMPLICIT_CO_CHURN
        assert edge.weight >= 0.0
        assert "co_churn_count" in edge.metadata


def test_git_miner_churn_score() -> None:
    miner = GitChurnMiner(".")
    # Record synthetic commits for testing math
    miner._record_commit({"src/auth.py", "src/user.py"})
    miner._record_commit({"src/auth.py", "src/user.py"})
    miner._record_commit({"src/auth.py", "src/token.py"})

    assert miner.file_commit_counts["src/auth.py"] == 3
    assert miner.co_change_counts[("src/auth.py", "src/user.py")] == 2

    score = miner.get_module_churn_score("src/auth.py")
    assert score > 0.0

    edges = miner.build_co_churn_edges(min_threshold=0.1)
    pair_sources = {(e.source, e.target) for e in edges}
    assert ("src/auth.py", "src/user.py") in pair_sources
