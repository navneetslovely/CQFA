# Campaign launchers

Each launcher runs one experiment group end to end and writes into
`results/<group>/`. They are thin wrappers over the module commands documented
in `REPRODUCE.md`, provided so a group can be reproduced with a single call.

Run from the archive root:

    bash scripts/campaigns/01_readout_basis.sh

Runtimes vary from minutes to hours; see the table at the end of `REPRODUCE.md`.
