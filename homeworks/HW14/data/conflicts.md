# Merge Conflicts

A merge conflict occurs when Git cannot automatically resolve differences between branches.

This usually happens when:
- the same line was modified in both branches
- one branch deletes a file while another modifies it

Git marks conflicts in the file and requires manual resolution.

After resolving:
git add .
git commit

Understanding conflicts is important for collaboration.