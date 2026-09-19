# Archived raw-fit tarballs (untracked, still in history)

The `*.tar.gz` raw-fit archives that used to live in the ten directories
below were untracked from HEAD (snag `brm-binaries-in-93881f23`) because
they are incompressible binaries — 43 files, ~351 MB — that every clone
and every linked worktree re-paid in full. They are NOT deleted: the bytes
remain in git history, and the `manifest.json` files beside them remain
the per-archive / per-member SHA-256 authority.

- `research/air_total_effects/results/fits/`
- `research/pupil_builtin_totals/results/fits/`
- `research/pupil_scale_totals/results/fits/`
- `research/pupil_total_effects/results/student_mixture/ordinary_cp/` (`source.tar.gz`)
- `research/pupil_total_effects/results/student_mixture/s2z_auto/native/` (`precursor.tar.gz`)
- `research/pupil_total_effects/results/student_mixture/s2z_auto/whmc/` (`source.tar.gz`)
- `research/pupil_total_effects/results/student_mixture/s2z_cp/whmc/` (`source.tar.gz`)
- `research/pupil_total_effects/results/student_mixture/s2z_ncp/whmc/` (`source.tar.gz`)
- `research/pupil_total_effects/results/student_mixture/total_cp/` (`source.tar.gz`)
- `research/rbest_centering/results/fits/`

To retrieve an archive, read it from the last commit that tracked it:

```sh
git show 3539d87644f7e8fe05853a00e4a4f7e8034ec594:<path-inside-repo> > <local-path>
```

or fetch it from the GitHub mirror at that SHA
(`https://raw.githubusercontent.com/nsiccha/BayesianRegressionModels.jl/3539d87644f7e8fe05853a00e4a4f7e8034ec594/<path-inside-repo>`).
`git log --all -- <path>` finds the same bytes if the pin above ever moves.

Do NOT re-commit fit artifacts to these directories: `.gitignore` now
blocks `research/**/*.tar.gz`, `research/**/*.jls{,.gz}`, and
`research/**/*.csv.gz`. Keep new archives in the producing run's scratch
or artifact store.
