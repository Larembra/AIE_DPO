# Git Rebase

Git rebase is used to move or reapply commits from one branch onto another base branch.

Instead of creating a merge commit, rebase rewrites the commit history.

Example:
git checkout feature-login
git rebase main

This takes the commits from feature-login and reapplies them on top of main.

Rebase creates a cleaner, linear history but can be dangerous if used on shared branches.