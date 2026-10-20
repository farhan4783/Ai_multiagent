"""
Automated Git Worktree Lifecycle Manager for Parallel Agent Sandboxes.

Programmatically creates, tracks, validates, rebases, merges, and tears down
isolated Git worktree environments (.worktrees/agent-<id>-task-<task_id>/).
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import os
import shutil
import subprocess
import time

from project_marl.core.logging import setup_logger

logger = setup_logger("worktree_manager")


@dataclass
class WorktreeInfo:
    """Metadata describing an active sandboxed Git worktree."""
    agent_id: str
    task_id: str
    branch_name: str
    worktree_path: str
    base_branch: str
    created_at: float = field(default_factory=time.time)

    def exists(self) -> bool:
        return os.path.exists(self.worktree_path)


class GitWorktreeManager:
    """
    Manages isolated Git worktrees enabling dozens of parallel coding agents
    to execute modifications on isolated branches without repository collisions.
    """

    def __init__(
        self,
        repo_dir: Optional[str] = None,
        worktree_root: Optional[str] = None,
    ) -> None:
        self.repo_dir = Path(repo_dir).resolve() if repo_dir else Path.cwd().resolve()
        self.worktree_root = (
            Path(worktree_root).resolve()
            if worktree_root
            else self.repo_dir / ".worktrees"
        )
        self.worktree_root.mkdir(parents=True, exist_ok=True)
        # Key: (agent_id, task_id) -> WorktreeInfo
        self._worktrees: Dict[Tuple[str, str], WorktreeInfo] = {}

    def _run_git(
        self,
        args: List[str],
        cwd: Optional[Path] = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Runs a git command with stdin=DEVNULL and UTF-8 encoding."""
        work_dir = str(cwd or self.repo_dir)
        cmd = ["git", "-C", work_dir] + args
        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=check,
                encoding="utf-8",
                errors="replace",
                stdin=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as e:
            logger.error("Git error running %s: stdout=%s stderr=%s", cmd, e.stdout, e.stderr)
            raise

    def is_git_repo(self) -> bool:
        """Verifies repository directory is a valid git repo."""
        res = self._run_git(["rev-parse", "--is-inside-work-tree"], check=False)
        return res.returncode == 0

    def get_current_branch(self) -> str:
        """Returns the current branch name of the base repo."""
        res = self._run_git(["rev-parse", "--abbrev-ref", "HEAD"])
        return res.stdout.strip()

    def create_worktree(
        self,
        agent_id: str,
        task_id: str,
        base_branch: Optional[str] = None,
    ) -> WorktreeInfo:
        """
        Creates an isolated worktree and branch for the agent task.

        Path: .worktrees/agent-<agent_id>-task-<task_id>/
        Branch: agent/<agent_id>/<task_id>
        """
        key = (agent_id, task_id)
        if key in self._worktrees and self._worktrees[key].exists():
            return self._worktrees[key]

        safe_name = f"agent-{agent_id}-task-{task_id}".replace("/", "_")
        wt_dir = self.worktree_root / safe_name
        branch_name = f"agent/{agent_id}/{task_id}"

        if not self.is_git_repo():
            wt_dir.mkdir(parents=True, exist_ok=True)
            info = WorktreeInfo(
                agent_id=agent_id,
                task_id=task_id,
                branch_name=branch_name,
                worktree_path=str(wt_dir),
                base_branch=base_branch or "main",
            )
            self._worktrees[key] = info
            return info

        base = base_branch or self.get_current_branch()

        # Prune dead worktree references and clean existing directory
        self._run_git(["worktree", "prune"], check=False)
        if wt_dir.exists():
            self._run_git(["worktree", "remove", "--force", str(wt_dir)], check=False)
            self._force_remove_directory(wt_dir)

        # Delete branch if it exists from previous run
        self._run_git(["branch", "-D", branch_name], check=False)

        # Add worktree (-B creates or resets branch)
        self._run_git([
            "worktree", "add",
            "-B", branch_name,
            str(wt_dir),
            base,
        ])

        info = WorktreeInfo(
            agent_id=agent_id,
            task_id=task_id,
            branch_name=branch_name,
            worktree_path=str(wt_dir),
            base_branch=base,
        )
        self._worktrees[key] = info
        logger.info("Created Git worktree at %s on branch %s", wt_dir, branch_name)
        return info

    def remove_worktree(
        self,
        agent_id: str,
        task_id: str,
        delete_branch: bool = True,
    ) -> bool:
        """
        Safely removes a worktree and optionally deletes the branch.
        """
        key = (agent_id, task_id)
        info = self._worktrees.pop(key, None)
        if not info:
            return False

        wt_path = Path(info.worktree_path)

        # Remove via git
        self._run_git(["worktree", "remove", "--force", str(wt_path)], check=False)

        # Ensure directory is purged on disk
        if wt_path.exists():
            self._force_remove_directory(wt_path)

        # Prune git internal worktree references
        self._run_git(["worktree", "prune"], check=False)

        if delete_branch:
            self._run_git(["branch", "-D", info.branch_name], check=False)

        logger.info("Removed Git worktree %s", info.worktree_path)
        return True

    def commit_in_worktree(
        self,
        agent_id: str,
        task_id: str,
        message: str,
    ) -> Optional[str]:
        """
        Stages all changes and commits them within the specified agent worktree.

        Returns:
            Commit hash if commit was created, None if no changes to commit.
        """
        info = self.get_worktree(agent_id, task_id)
        if not info:
            raise ValueError(f"Worktree for agent={agent_id} task={task_id} not found.")

        wt_dir = Path(info.worktree_path)

        # Check dirty status
        status = self._run_git(["status", "--porcelain"], cwd=wt_dir)
        if not status.stdout.strip():
            logger.info("No changes to commit in worktree %s", info.worktree_path)
            return None

        self._run_git(["add", "-A"], cwd=wt_dir)
        self._run_git(["commit", "-m", message], cwd=wt_dir)

        rev = self._run_git(["rev-parse", "HEAD"], cwd=wt_dir)
        commit_sha = rev.stdout.strip()
        logger.info("Committed %s in worktree %s", commit_sha, info.worktree_path)
        return commit_sha

    def can_merge_cleanly(
        self,
        agent_id: str,
        task_id: str,
        target_branch: Optional[str] = None,
    ) -> bool:
        """
        Verifies whether the agent worktree branch can be cleanly merged into target
        without conflicts using git merge-tree.
        """
        info = self.get_worktree(agent_id, task_id)
        if not info:
            return False

        target = target_branch or info.base_branch

        # Test merge compatibility using git merge-tree
        res = self._run_git(["merge-tree", target, info.branch_name], check=False)
        if res.returncode != 0:
            return False

        # In git merge-tree, conflict markers <<<<<<< indicate collisions
        return "<<<<<<<" not in res.stdout

    def merge_worktree_into_base(
        self,
        agent_id: str,
        task_id: str,
        target_branch: Optional[str] = None,
        commit_message: Optional[str] = None,
    ) -> bool:
        """
        Merges worktree branch into base repository branch.
        """
        info = self.get_worktree(agent_id, task_id)
        if not info:
            return False

        target = target_branch or info.base_branch
        msg = commit_message or f"Merge branch '{info.branch_name}' for Task {task_id}"

        # In base repo, switch to target and merge
        try:
            self._run_git(["checkout", target])
            self._run_git(["merge", "--no-ff", info.branch_name, "-m", msg])
            logger.info("Successfully merged %s into %s", info.branch_name, target)
            return True
        except subprocess.CalledProcessError:
            self._run_git(["merge", "--abort"], check=False)
            logger.warning("Merge failed for %s into %s", info.branch_name, target)
            return False

    def get_worktree(self, agent_id: str, task_id: str) -> Optional[WorktreeInfo]:
        """Returns worktree metadata for given agent task."""
        return self._worktrees.get((agent_id, task_id))

    def list_active_worktrees(self) -> List[WorktreeInfo]:
        """Lists all tracked active worktrees."""
        return list(self._worktrees.values())

    def teardown_all(self) -> int:
        """Tears down all active worktrees and prunes references."""
        keys = list(self._worktrees.keys())
        for agent_id, task_id in keys:
            self.remove_worktree(agent_id, task_id, delete_branch=True)
        self._run_git(["worktree", "prune"], check=False)
        return len(keys)

    @staticmethod
    def _force_remove_directory(path: Path) -> None:
        """Robust directory removal on Windows handling read-only git pack files."""
        def handle_remove_readonly(func: Any, sub_path: str, exc: Any) -> None:
            try:
                os.chmod(sub_path, 0o777)
                func(sub_path)
            except OSError:
                pass

        if path.exists():
            shutil.rmtree(str(path), onerror=handle_remove_readonly)
