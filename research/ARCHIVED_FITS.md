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
blocks `research/**/*.tar.gz`, `research/**/*.jls{,.gz}`,
`research/**/*.csv.gz`, `research/**/*.rds.gz`, `research/**/*.tsv.gz`,
and the `*_pairs.{tsv,aov.json,png}` / `selection_losses.tsv` /
`ppc_intervals.tsv` result classes (see the 2026-09-28 migration below).
Keep new archives in the producing run's scratch or artifact store.


## KB capsule retrieval (migrated fit archives)

All 43 archives above are also stored as verified KB-upload
capsules, chunked at 20 MB. `parts` live in the studies'
`manifest.json` files (same schema, `parts: [{index, path,
bytes, sha256}]`); re-download with `GET /code?path=<part>&raw=1`,
concatenate in `index` order, and SHA-256 the result against the
`sha256` beside it. The full part list:

- `research/air_total_effects/results/fits/air-brms-whmc-cluster-independent-v1.tar.gz` (4473854 B, sha256 `825d9a1f08ec130a30e8b06954e45f3ebc9f6d46b7cfa20b6a677544f8be2b41`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/36431e06c39fe134.bin` (4473854 B, sha256 `825d9a1f08ec130a30e8b06954e45f3ebc9f6d46b7cfa20b6a677544f8be2b41`)
- `research/air_total_effects/results/fits/air-brms-whmc-cluster-intercept-v1.tar.gz` (2724318 B, sha256 `48f295e3f8e38ac8cbc6d54073f8231089442f99d2707ac99794351051838226`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/e4ac53d142483a56.bin` (2724318 B, sha256 `48f295e3f8e38ac8cbc6d54073f8231089442f99d2707ac99794351051838226`)
- `research/air_total_effects/results/fits/air-native-cluster-independent-v1-ordinary_ncp.tar.gz` (2181649 B, sha256 `72fe4057d285b37f269f7f62fddff6b22d14dbb3bdae837655b239aa4f9a72d9`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/682273fad71e44ba.bin` (2181649 B, sha256 `72fe4057d285b37f269f7f62fddff6b22d14dbb3bdae837655b239aa4f9a72d9`)
- `research/air_total_effects/results/fits/air-native-cluster-independent-v1-s2z_auto.tar.gz` (4022032 B, sha256 `eda68991c30ddba96da12c445a40f88ea1095b97370b944709f9d0f8262ae384`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/c74cd251d0e20c19.bin` (4022032 B, sha256 `eda68991c30ddba96da12c445a40f88ea1095b97370b944709f9d0f8262ae384`)
- `research/air_total_effects/results/fits/air-native-cluster-independent-v1-s2z_cp.tar.gz` (3242464 B, sha256 `49af53e301b62a6495bdfe90c957f8138e5dd9fc03c636bbbcb257239334e1d0`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/07e75096827497a0.bin` (3242464 B, sha256 `49af53e301b62a6495bdfe90c957f8138e5dd9fc03c636bbbcb257239334e1d0`)
- `research/air_total_effects/results/fits/air-native-cluster-independent-v1-s2z_ncp.tar.gz` (3243119 B, sha256 `b0d1b437748f00a84564cbdc343726943e83b64d01eb1a4d55c0ab746c2bfd99`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/40a2e2beeb6607dd.bin` (3243119 B, sha256 `b0d1b437748f00a84564cbdc343726943e83b64d01eb1a4d55c0ab746c2bfd99`)
- `research/air_total_effects/results/fits/air-native-cluster-intercept-v1-ordinary_ncp.tar.gz` (1429885 B, sha256 `2db41a44f28e089e800bc281f2ad5b7a2cabaf55a8b7cd53ff1ea5f2d2cd960e`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/9523cd1389f48078.bin` (1429885 B, sha256 `2db41a44f28e089e800bc281f2ad5b7a2cabaf55a8b7cd53ff1ea5f2d2cd960e`)
- `research/air_total_effects/results/fits/air-native-cluster-intercept-v1-s2z_auto.tar.gz` (2635825 B, sha256 `1d0c57723575fa92ffbf53f78343979c4b1ce2833d5f853a25b66996d69f9de5`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/90822b7a14db21d7.bin` (2635825 B, sha256 `1d0c57723575fa92ffbf53f78343979c4b1ce2833d5f853a25b66996d69f9de5`)
- `research/air_total_effects/results/fits/air-native-cluster-intercept-v1-s2z_cp.tar.gz` (2150186 B, sha256 `e01ea57b993e69daacee0a19c2bccb1bf9ae813000e9a1ffc57bc7502297108f`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/d3754dac5ad8082a.bin` (2150186 B, sha256 `e01ea57b993e69daacee0a19c2bccb1bf9ae813000e9a1ffc57bc7502297108f`)
- `research/air_total_effects/results/fits/air-native-cluster-intercept-v1-s2z_ncp.tar.gz` (2157098 B, sha256 `ab43268012a20a6647fce60e3f4369ccb1ffa6ee2d168efe863c81bc6d82d559`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/00d660f785082633.bin` (2157098 B, sha256 `ab43268012a20a6647fce60e3f4369ccb1ffa6ee2d168efe863c81bc6d82d559`)
- `research/air_total_effects/results/fits/air-summary-cluster-independent-v1.tar.gz` (1760902 B, sha256 `f4d8723907778a2b0896b090902377c5905e218951e62bc367efb998b57e460f`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/764b1186b3cbe0b5.bin` (1760902 B, sha256 `f4d8723907778a2b0896b090902377c5905e218951e62bc367efb998b57e460f`)
- `research/air_total_effects/results/fits/air-summary-cluster-intercept-v1.tar.gz` (1046189 B, sha256 `6b6747731cf7ec0a85fe0948c0d2afe1cec1e54195d3f1c73a55b0267abd4f40`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/ba8143d8efb9b6cd.bin` (1046189 B, sha256 `6b6747731cf7ec0a85fe0948c0d2afe1cec1e54195d3f1c73a55b0267abd4f40`)
- `research/air_total_effects/results/fits/air-totals-cluster-independent-v1.tar.gz` (12635971 B, sha256 `9b1df40d8d64428f06b207611ef38af65e777df6feda3434f3a3bc417d8194f0`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/eb085b370871fd08.bin` (12635971 B, sha256 `9b1df40d8d64428f06b207611ef38af65e777df6feda3434f3a3bc417d8194f0`)
- `research/air_total_effects/results/fits/air-totals-cluster-intercept-v1.tar.gz` (7697032 B, sha256 `890750abc567d5bf71b6619bd75fd54ff893b817b58ecbe75f44f1c2293ea16c`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/7b3c19dcbe1d15a3.bin` (7697032 B, sha256 `890750abc567d5bf71b6619bd75fd54ff893b817b58ecbe75f44f1c2293ea16c`)
- `research/pupil_builtin_totals/results/fits/pupil3-builtin-brms-summary-v1.tar.gz` (6119183 B, sha256 `668a230e222a6d383ffd6f7266ea30b9aecb870d0458357fa840e4b64e0ee922`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/784e7c4d55f2fbf1.bin` (6119183 B, sha256 `668a230e222a6d383ffd6f7266ea30b9aecb870d0458357fa840e4b64e0ee922`)
- `research/pupil_builtin_totals/results/fits/pupil3-builtin-totals-v2.tar.gz` (29827042 B, sha256 `2f8db847f019e64dccc229dd26336ce6d02e86f13a8c2540345afdd3a4684209`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/03a413c3ff96387c.bin` (20971520 B, sha256 `80d39aa3d80b34b3d60b6dbd2f957b14f415f02375938265cdcb7e8039aa899b`)
  - part 1: `/home/niko/.local/state/kb-agents/uploads/f554441dec7772c8.bin` (8855522 B, sha256 `7990fd04793df2175659b6d7b4c08b9ff29c05ad0f0f4c360f0e6cd3d7de305f`)
- `research/pupil_scale_totals/results/fits/native-ordinary_ncp.tar.gz` (7762505 B, sha256 `c35e0fc7f1ce7e14185ad2bf600c45ed22d2cc303d4a5c09cc67c5e72ab651e9`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/ec64f5d8c07ea395.bin` (7762505 B, sha256 `c35e0fc7f1ce7e14185ad2bf600c45ed22d2cc303d4a5c09cc67c5e72ab651e9`)
- `research/pupil_scale_totals/results/fits/native-s2z_auto.tar.gz` (14435628 B, sha256 `6392e4e8e22cdaa369d4e9fdbc606ce537946971d954a38fc570f21395b03c65`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/ad1af78118737879.bin` (14435628 B, sha256 `6392e4e8e22cdaa369d4e9fdbc606ce537946971d954a38fc570f21395b03c65`)
- `research/pupil_scale_totals/results/fits/native-s2z_cp.tar.gz` (11469377 B, sha256 `23b8d65d5055a739b87c9b7d0944d51f1e1d2131d1d09756ff13aedc66c22647`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/f42d0277b7b1d9a3.bin` (11469377 B, sha256 `23b8d65d5055a739b87c9b7d0944d51f1e1d2131d1d09756ff13aedc66c22647`)
- `research/pupil_scale_totals/results/fits/native-s2z_ncp.tar.gz` (11475707 B, sha256 `b66382e41ab101ca50de4960b400aaf6ec72b3061ce1dbbc0965c83f16f253f1`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/5d725289dafc7921.bin` (11475707 B, sha256 `b66382e41ab101ca50de4960b400aaf6ec72b3061ce1dbbc0965c83f16f253f1`)
- `research/pupil_scale_totals/results/fits/pupil4-brms-whmc-v1.tar.gz` (32109507 B, sha256 `e5d29c25b444056c863ab54aa5367a64b510a657d3db995983e3de674babecc5`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/c21f441aac1795a9.bin` (20971520 B, sha256 `2d9639b8088e64ea818a0568fd5aac8c97d35452d8ce18bcc6889e9a7c30459c`)
  - part 1: `/home/niko/.local/state/kb-agents/uploads/2d52ca1f1f8b7b22.bin` (11137987 B, sha256 `358696a8a2fb9db47f66c093ba0ce6b417246a0d87877bf62b14179c45fe7a98`)
- `research/pupil_scale_totals/results/fits/pupil4-summary-v1.tar.gz` (6867814 B, sha256 `34e8da736e79033e65b7ae1822cb183ae2109527c0b44edfbf4f073f513ed544`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/3d1b2504f14e877e.bin` (6867814 B, sha256 `34e8da736e79033e65b7ae1822cb183ae2109527c0b44edfbf4f073f513ed544`)
- `research/pupil_scale_totals/results/fits/pupil4-totals-v1.tar.gz` (43425587 B, sha256 `59b8620717fac028d36b97cafad92f8ec0510c00100ad420f6bb878c67cc7a9c`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/6abdbd84d971856a.bin` (20971520 B, sha256 `17980d5602684e0b73f1f9f65e524902923c0731a8a4d7bbd023c699ce99ecff`)
  - part 1: `/home/niko/.local/state/kb-agents/uploads/a046d59e7fce45bd.bin` (20971520 B, sha256 `b17b78072652b84d9e2f0af40054cbaf614f160eb2428c9fe0427d23dd092fdd`)
  - part 2: `/home/niko/.local/state/kb-agents/uploads/adec912a7f543525.bin` (1482547 B, sha256 `ed76739cc2fba03107575c1cff7ad1152281f29c4cf4b09c0950dd549f5ed642`)
- `research/pupil_total_effects/results/student_mixture/ordinary_cp/source.tar.gz` (82165 B, sha256 `7f77ced91249a57b2304f526c5599b005e83a1c2d191187feedb7fc51ed5035d`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/9d6a8201fe623cbe.bin` (82165 B, sha256 `7f77ced91249a57b2304f526c5599b005e83a1c2d191187feedb7fc51ed5035d`)
- `research/pupil_total_effects/results/student_mixture/s2z_auto/native/precursor.tar.gz` (1801194 B, sha256 `7c6e657472ce09e1ea3765201b936475ff21954a2ad039d7a049377ae6891235`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/031e3ad7c04cec1e.bin` (1801194 B, sha256 `7c6e657472ce09e1ea3765201b936475ff21954a2ad039d7a049377ae6891235`)
- `research/pupil_total_effects/results/student_mixture/s2z_auto/whmc/source.tar.gz` (83994 B, sha256 `b668cce306aba448b011141ee45f0efd8e3e5bf7dbce9a6afaec12f0c8aef9e5`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/06453619163aff55.bin` (83994 B, sha256 `b668cce306aba448b011141ee45f0efd8e3e5bf7dbce9a6afaec12f0c8aef9e5`)
- `research/pupil_total_effects/results/student_mixture/s2z_cp/whmc/source.tar.gz` (83992 B, sha256 `2d05e8731ec58f7248a8ab22e53818445281aeaf05959d7cbb8f1b263c1a60be`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/61300f66a6060a42.bin` (83992 B, sha256 `2d05e8731ec58f7248a8ab22e53818445281aeaf05959d7cbb8f1b263c1a60be`)
- `research/pupil_total_effects/results/student_mixture/s2z_ncp/whmc/source.tar.gz` (83992 B, sha256 `f370c8e62c106a8587c861c19dfbf8407ba29dbc43cd0849d3ce1ea26bfbfbb3`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/b5f98d47ae1e6eea.bin` (83992 B, sha256 `f370c8e62c106a8587c861c19dfbf8407ba29dbc43cd0849d3ce1ea26bfbfbb3`)
- `research/pupil_total_effects/results/student_mixture/total_cp/source.tar.gz` (80691 B, sha256 `995490745eb0da5ea0ca74d84ed73429687b0f207253b467ca110cf29729da14`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/66f07129a6eca097.bin` (80691 B, sha256 `995490745eb0da5ea0ca74d84ed73429687b0f207253b467ca110cf29729da14`)
- `research/rbest_centering/results/fits/rbest-matrix-AS-v1.tar.gz` (54338562 B, sha256 `01bc898d1c59b3260aaaff85011c330b5d64ef1c9eb0f52204dff48095f9234c`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/4e608ef53531806c.bin` (20971520 B, sha256 `ecd271e020bbcadf5a2022fee6d5d50cf28d34a0999e98eef2d7863d754da0e3`)
  - part 1: `/home/niko/.local/state/kb-agents/uploads/6e82cf75e01a0fdc.bin` (20971520 B, sha256 `739b59dfbddff9284eb5c69d45c0ec55c3743dbc2d08c6d32b3c517500b5ed19`)
  - part 2: `/home/niko/.local/state/kb-agents/uploads/eceef485cd05801f.bin` (12395522 B, sha256 `74f23d3c59e314f0d4a0f69fc23b5d28d1388b031b8ac4135950d5df5a29e756`)
- `research/rbest_centering/results/fits/rbest-matrix-crohn-v1.tar.gz` (42893107 B, sha256 `d62ac02f8aa89f83294765303d51c01397cc1b02aa13f03906cab4dcc5e3785e`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/24543f0f73e997d2.bin` (20971520 B, sha256 `1ef14ae95c21437483b610760f9bd07c7ee34ab7a19740da38cf7654310d0ac7`)
  - part 1: `/home/niko/.local/state/kb-agents/uploads/2ed3e0d0038ee559.bin` (20971520 B, sha256 `5fdf0b44b6f306dacec94c2ca48619c4566e94161a55cea2c264f1b1fabe1ac7`)
  - part 2: `/home/niko/.local/state/kb-agents/uploads/ac03ae1b765f2d43.bin` (950067 B, sha256 `7697a83c12b50f3a83bc3cf7d09422beb3783bbf0194debf7c5aebfdcdd91f7c`)
- `research/rbest_centering/results/fits/rbest-native-AS-v1-rbest_cp.tar.gz` (4734830 B, sha256 `9be66f317d0b018441d6b9c420cd5082eedf109eeb37683199de497e49fe7dfe`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/189310869b26e71b.bin` (4734830 B, sha256 `9be66f317d0b018441d6b9c420cd5082eedf109eeb37683199de497e49fe7dfe`)
- `research/rbest_centering/results/fits/rbest-native-AS-v1-rbest_ncp.tar.gz` (5007475 B, sha256 `cd385f4441c15acfb643c27d0b7d3dda8dc214ddc56712556a0b74d91a335213`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/cc9932f4945fe59c.bin` (5007475 B, sha256 `cd385f4441c15acfb643c27d0b7d3dda8dc214ddc56712556a0b74d91a335213`)
- `research/rbest_centering/results/fits/rbest-native-AS-v1-s2z_cp.tar.gz` (4993397 B, sha256 `c96239f451f83640f132969954128e2693913f500e49711b8cd38a2c8897b6fc`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/daaffa943427f7bb.bin` (4993397 B, sha256 `c96239f451f83640f132969954128e2693913f500e49711b8cd38a2c8897b6fc`)
- `research/rbest_centering/results/fits/rbest-native-AS-v1-s2z_ncp.tar.gz` (5004204 B, sha256 `357be41b8eb9cc8ec7014418b0bb2638e998ab5d0910d9b4df2cc737aecd70db`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/85de5451d2bb0971.bin` (5004204 B, sha256 `357be41b8eb9cc8ec7014418b0bb2638e998ab5d0910d9b4df2cc737aecd70db`)
- `research/rbest_centering/results/fits/rbest-native-AS-v1-stan_cp.tar.gz` (4394087 B, sha256 `ca1a7970a65aaf0c9ff10cf76d5a42019a7d475f44563d3b0c860cb5fb110846`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/c81553cac2204601.bin` (4394087 B, sha256 `ca1a7970a65aaf0c9ff10cf76d5a42019a7d475f44563d3b0c860cb5fb110846`)
- `research/rbest_centering/results/fits/rbest-native-AS-v1-stan_ncp.tar.gz` (4703773 B, sha256 `290977cd38390f5ade47edb59c1d11f56922b9f8355fe9db9dac694736685a97`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/b5b7913effea5ca4.bin` (4703773 B, sha256 `290977cd38390f5ade47edb59c1d11f56922b9f8355fe9db9dac694736685a97`)
- `research/rbest_centering/results/fits/rbest-native-crohn-v1-rbest_cp.tar.gz` (4201353 B, sha256 `ba719e7f02a88cfc5e59619956413a61c5a47e6d9a677525cbb1f493018afdda`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/06138a1f83a70937.bin` (4201353 B, sha256 `ba719e7f02a88cfc5e59619956413a61c5a47e6d9a677525cbb1f493018afdda`)
- `research/rbest_centering/results/fits/rbest-native-crohn-v1-rbest_ncp.tar.gz` (4207047 B, sha256 `1c4444ae74049d71b46896413dc2d991d16797c118e6e173c57a017ffea13f8d`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/c39ade6d8d0524cd.bin` (4207047 B, sha256 `1c4444ae74049d71b46896413dc2d991d16797c118e6e173c57a017ffea13f8d`)
- `research/rbest_centering/results/fits/rbest-native-crohn-v1-s2z_cp.tar.gz` (4160010 B, sha256 `6eff1d28a471d3fc2a0bce60780859c5978063ff15641d1ad16f6b30db99683d`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/a953d2bc5b876cf8.bin` (4160010 B, sha256 `6eff1d28a471d3fc2a0bce60780859c5978063ff15641d1ad16f6b30db99683d`)
- `research/rbest_centering/results/fits/rbest-native-crohn-v1-s2z_ncp.tar.gz` (4192482 B, sha256 `49d7f12d15effb3e660d35dc6cc07c1c02a82b94d02d7bb31547f0ecf917664c`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/691f5739b1344dcf.bin` (4192482 B, sha256 `49d7f12d15effb3e660d35dc6cc07c1c02a82b94d02d7bb31547f0ecf917664c`)
- `research/rbest_centering/results/fits/rbest-native-crohn-v1-stan_cp.tar.gz` (3945641 B, sha256 `db4cb5e47cc2114e21e996932406330fe6eb0780f889b21fc83e0b7d86d482c1`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/e26420f055c6e8a3.bin` (3945641 B, sha256 `db4cb5e47cc2114e21e996932406330fe6eb0780f889b21fc83e0b7d86d482c1`)
- `research/rbest_centering/results/fits/rbest-native-crohn-v1-stan_ncp.tar.gz` (3973606 B, sha256 `1d86a863b578154ed2be9419d2c37f93921b9208e602829775e5d5c926c15b44`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/e3b3564db04cd962.bin` (3973606 B, sha256 `1d86a863b578154ed2be9419d2c37f93921b9208e602829775e5d5c926c15b44`)

## HEAD-only big-file migration (2026-09-28, todo `19vta7w`)

The 56 files ≥1 MB below (149.4 MB: serialized draws, pairs tables,
plot specs/figures, selection-loss tables) were untracked from HEAD and
migrated to verified KB-upload capsules (same `research/capsule_upload.py`
flow: 20 MB chunks, read-back SHA verify). Per-directory `manifest.json`
files carry `parts: [{index, path, bytes, sha256}]`; re-download with
`GET /code?path=<part>&raw=1`, concatenate in `index` order, and SHA-256
the result against the `sha256` beside it. Last tracked commit:

```sh
git show 3db139b0e6ee900915cd4802eefddc342d723b26:<path-inside-repo> > <local-path>
```

Small twins stay tracked (all ≤887 KB, some relatively linked from
research notes or GitHub blob-linked from docs pages): 18 small `*.jls.gz`,
3 small `*.rds.gz`, 3 small `*.tsv.gz`, all 8 `*.json.gz`, 10 small
`*_pairs.png`, 6 small `selection_losses.tsv`, 1 tiny `ppc_intervals.tsv`.
`.gitignore` blocks the migrated name classes for NEW files; tracked small
twins are unaffected. Research `plot.jl`/`prepare_*.jl` scripts regenerate
the pairs tables/figures from retrieved capsules; nothing under `docs/`,
`test/`, `src/`, or `examples/` reads the migrated files at build time.

- `research/air_total_effects/results/cluster-independent/s2z_pairs.aov.json` (2576410 B, sha256 `148f4f0e6244211cae7a278ca085a8b4824fea639f05f3da4cdaa2c022da1e49`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/546567ae2fbd5223.bin` (2576410 B, sha256 `148f4f0e6244211cae7a278ca085a8b4824fea639f05f3da4cdaa2c022da1e49`)
- `research/air_total_effects/results/cluster-independent/s2z_pairs.tsv` (1531619 B, sha256 `a58f39cded901639fea1f22fb4b60886b3b0d9dec81d156ec3a4523062cb14d4`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/6c390f3be27e5fce.bin` (1531619 B, sha256 `a58f39cded901639fea1f22fb4b60886b3b0d9dec81d156ec3a4523062cb14d4`)
- `research/air_total_effects/results/cluster-independent/total_pairs.aov.json` (2552577 B, sha256 `db5499fb99e09923051b824dcb99cdac2e770b49f47603cd09148af3106181ba`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/c0d5099ad9f75e62.bin` (2552577 B, sha256 `db5499fb99e09923051b824dcb99cdac2e770b49f47603cd09148af3106181ba`)
- `research/air_total_effects/results/cluster-independent/total_pairs.tsv` (1507786 B, sha256 `c35b8a0885c622be594c0aaae1a3a0f03df19be6053d1495dfac9406bf0ea0f1`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/f1a617b65e5f2973.bin` (1507786 B, sha256 `c35b8a0885c622be594c0aaae1a3a0f03df19be6053d1495dfac9406bf0ea0f1`)
- `research/air_total_effects/results/cluster-intercept/s2z_pairs.aov.json` (2603637 B, sha256 `334ba379d3f3ab0f5cfc9ba6086b4d0d9c09854eaccb7340539d37fb03004fce`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/543ac2c4093dc929.bin` (2603637 B, sha256 `334ba379d3f3ab0f5cfc9ba6086b4d0d9c09854eaccb7340539d37fb03004fce`)
- `research/air_total_effects/results/cluster-intercept/s2z_pairs.tsv` (1558846 B, sha256 `ada41940aef480a407cfca431ea144bbc873deab5c14d3d177d4cd1d30eae3d5`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/8075ff4ba6745b50.bin` (1558846 B, sha256 `ada41940aef480a407cfca431ea144bbc873deab5c14d3d177d4cd1d30eae3d5`)
- `research/air_total_effects/results/cluster-intercept/total_pairs.aov.json` (2551406 B, sha256 `7d2cd9a9e862c2c226c9d3b22b4ba3761b0b28cdbdaa620a8724a5f084bd367c`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/dfa877c1bcc348d5.bin` (2551406 B, sha256 `7d2cd9a9e862c2c226c9d3b22b4ba3761b0b28cdbdaa620a8724a5f084bd367c`)
- `research/air_total_effects/results/cluster-intercept/total_pairs.tsv` (1506615 B, sha256 `6ce67e055a6a924bb9bfd0acc02d097926af5d15a808304d70884339e2ebbac4`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/dc80f57dc7ea6cc5.bin` (1506615 B, sha256 `6ce67e055a6a924bb9bfd0acc02d097926af5d15a808304d70884339e2ebbac4`)
- `research/centering_refresh/results/radon/posthoc_gradient/selection_losses.tsv` (2621722 B, sha256 `e13db8380de8e18dc258fb8f3c91694fe42adf934d795e8f28f1e9e1da8f848d`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/11c436b2758982c8.bin` (2621722 B, sha256 `e13db8380de8e18dc258fb8f3c91694fe42adf934d795e8f28f1e9e1da8f848d`)
- `research/centering_refresh/results/radon/posthoc_position/selection_losses.tsv` (2653307 B, sha256 `3d1b426e8b42a86efb38ee5ba0cc42a1c3b4df4988ac3ad7d57760cb19c22790`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/fb131485ae8a9539.bin` (2653307 B, sha256 `3d1b426e8b42a86efb38ee5ba0cc42a1c3b4df4988ac3ad7d57760cb19c22790`)
- `research/pupil_builtin_totals/results/total_pairs.aov.json` (2371766 B, sha256 `c24b942e3fa34be78561bd46d590438c45e166ccd628f5ee7bdbe9f8430b7195`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/6944b04f3a35f1e8.bin` (2371766 B, sha256 `c24b942e3fa34be78561bd46d590438c45e166ccd628f5ee7bdbe9f8430b7195`)
- `research/pupil_builtin_totals/results/total_pairs.tsv` (1326975 B, sha256 `670c6f46c7c5afbd704091a8456c84bd99242a874229b38b71ac1a144062e2aa`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/d465076978bf4fd2.bin` (1326975 B, sha256 `670c6f46c7c5afbd704091a8456c84bd99242a874229b38b71ac1a144062e2aa`)
- `research/pupil_scale_totals/results/s2z_pairs.aov.json` (2413881 B, sha256 `cb0ec81fe7927c512a4879db0a1a9c9bc1a391a18ebc10e70e33caa8937e3ba5`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/331f064b94b50fb6.bin` (2413881 B, sha256 `cb0ec81fe7927c512a4879db0a1a9c9bc1a391a18ebc10e70e33caa8937e3ba5`)
- `research/pupil_scale_totals/results/s2z_pairs.tsv` (1369090 B, sha256 `3dcd0d85e288eff331f3fc990d9c2418cab835007ec05cc2f88c9784193df4ec`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/269b9858b955eaad.bin` (1369090 B, sha256 `3dcd0d85e288eff331f3fc990d9c2418cab835007ec05cc2f88c9784193df4ec`)
- `research/pupil_scale_totals/results/total_pairs.aov.json` (2370491 B, sha256 `f06e65a18ee1c6579f3845f6c9f81f88a67a769c06ac2622993097cda0c9f97a`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/0f47866d7df19892.bin` (2370491 B, sha256 `f06e65a18ee1c6579f3845f6c9f81f88a67a769c06ac2622993097cda0c9f97a`)
- `research/pupil_scale_totals/results/total_pairs.tsv` (1325700 B, sha256 `935663cacd5f1f42b21304f0cac79a3e8cdc2bb1633fca0b30ca88c8c68cab95`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/f5b82dfa6dcce155.bin` (1325700 B, sha256 `935663cacd5f1f42b21304f0cac79a3e8cdc2bb1633fca0b30ca88c8c68cab95`)
- `research/pupil_total_effects/results/gaussian/fits/brms_ncp.jls.gz` (2067652 B, sha256 `ab774d2cd1f71341b51637eaa4b42f82cd60f1494993915c3c8e527156088719`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/7de4c48a9820fdd0.bin` (2067652 B, sha256 `ab774d2cd1f71341b51637eaa4b42f82cd60f1494993915c3c8e527156088719`)
- `research/pupil_total_effects/results/gaussian/native_ncp/native_ncp.jls.gz` (1378146 B, sha256 `e4213d1aff93fdbd52dc5ff49b8af0279b1c2e79a96eefad27f2c4c1107522fc`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/0f84e8337d185321.bin` (1378146 B, sha256 `e4213d1aff93fdbd52dc5ff49b8af0279b1c2e79a96eefad27f2c4c1107522fc`)
- `research/pupil_total_effects/results/gaussian/native_ncp/pupil-1.csv.gz` (2645107 B, sha256 `f98ce95a7c7f78c63e02b20b426f21f17633dfa9fcd7bd2321707edc3474bba9`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/7c59a613dd69cc5e.bin` (2645107 B, sha256 `f98ce95a7c7f78c63e02b20b426f21f17633dfa9fcd7bd2321707edc3474bba9`)
- `research/pupil_total_effects/results/gaussian/native_ncp/sampling.tsv.gz` (1600367 B, sha256 `7bc469f9f67e2008edbad13c09f66df02facf175ab0561478095075e33519832`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/0bd973363f0e26a8.bin` (1600367 B, sha256 `7bc469f9f67e2008edbad13c09f66df02facf175ab0561478095075e33519832`)
- `research/pupil_total_effects/results/online_adaptation/gaussian_gradient/checkpoint_final.jls.gz` (2062549 B, sha256 `7773a82aac4b1698ef5ae29bcedbbefdff4c16b3e19f2ee46bf68a574b6656c4`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/c8e1894401ef25dc.bin` (2062549 B, sha256 `7773a82aac4b1698ef5ae29bcedbbefdff4c16b3e19f2ee46bf68a574b6656c4`)
- `research/pupil_total_effects/results/online_adaptation/gaussian_position/checkpoint_final.jls.gz` (2061870 B, sha256 `ba8356d6d4d0835683536f4aa677bf67e242f0da072c5fda591d85afdb1dd684`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/90a39cb94d121108.bin` (2061870 B, sha256 `ba8356d6d4d0835683536f4aa677bf67e242f0da072c5fda591d85afdb1dd684`)
- `research/pupil_total_effects/results/online_adaptation/gaussian_posthoc_gradient/checkpoint_final.jls.gz` (2039621 B, sha256 `2397de8b6df7e7249a8d05f8f7650eb361ea77144a6194e07351e3f926a598d1`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/29c630cdb1b26276.bin` (2039621 B, sha256 `2397de8b6df7e7249a8d05f8f7650eb361ea77144a6194e07351e3f926a598d1`)
- `research/pupil_total_effects/results/online_adaptation/student_gradient/checkpoint_final.jls.gz` (2088468 B, sha256 `1f364057d1027564edb9885bb9dbd4feb8aaba545f0c18dd0d28a45988c5fe9f`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/5bab304c399f448b.bin` (2088468 B, sha256 `1f364057d1027564edb9885bb9dbd4feb8aaba545f0c18dd0d28a45988c5fe9f`)
- `research/pupil_total_effects/results/online_adaptation/student_position/checkpoint_final.jls.gz` (2091715 B, sha256 `03f87f95920a89d8182446359a73936a79229f16819e87c0445c3e3870ef51ec`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/28f6cd7b33623f5a.bin` (2091715 B, sha256 `03f87f95920a89d8182446359a73936a79229f16819e87c0445c3e3870ef51ec`)
- `research/pupil_total_effects/results/online_adaptation/student_posthoc_gradient/checkpoint_final.jls.gz` (2108120 B, sha256 `199cb6cc77045b5304e1bb06e4e04d9b4a74d17b7ff05951dbb139756c8705e9`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/4eb4e138545ff2a6.bin` (2108120 B, sha256 `199cb6cc77045b5304e1bb06e4e04d9b4a74d17b7ff05951dbb139756c8705e9`)
- `research/pupil_total_effects/results/student_mixture/fits/brms_ncp.jls.gz` (2087853 B, sha256 `c1b4cf4fa407408042af70d62ab433be5c20f5256c07afa2697462201ed96f9f`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/0b147826f260488a.bin` (2087853 B, sha256 `c1b4cf4fa407408042af70d62ab433be5c20f5256c07afa2697462201ed96f9f`)
- `research/pupil_total_effects/results/student_mixture/native_ncp/native_ncp.jls.gz` (1378241 B, sha256 `0ad4978adee387a72a938035a0d7014ebbe6dcc8cb4f93f5f671a2f846c37fee`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/ef7b165efa6aeac3.bin` (1378241 B, sha256 `0ad4978adee387a72a938035a0d7014ebbe6dcc8cb4f93f5f671a2f846c37fee`)
- `research/pupil_total_effects/results/student_mixture/native_ncp/pupil-1.csv.gz` (2648150 B, sha256 `d00dbec9b4cc99a12cff0bc2b119b6f19f0281afe78cebccb828dd6e1f258a9e`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/6169d5e637a62e00.bin` (2648150 B, sha256 `d00dbec9b4cc99a12cff0bc2b119b6f19f0281afe78cebccb828dd6e1f258a9e`)
- `research/pupil_total_effects/results/student_mixture/native_ncp/sampling.tsv.gz` (1600658 B, sha256 `600915914dff5aa460d9e31489e4ef5272dfb22890896087251c16a9d6ca0587`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/458c0722ebe9ee3e.bin` (1600658 B, sha256 `600915914dff5aa460d9e31489e4ef5272dfb22890896087251c16a9d6ca0587`)
- `research/pupil_total_effects/results/student_mixture/ordinary_cp/brms_cp.jls.gz` (2078585 B, sha256 `f8909b374769c68e6268c87ce6a4d044050875adaca014af30551cc0687be370`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/539e03a01d9a4e8a.bin` (2078585 B, sha256 `f8909b374769c68e6268c87ce6a4d044050875adaca014af30551cc0687be370`)
- `research/pupil_total_effects/results/student_mixture/s2z_auto/native/fit.jls.gz` (4360132 B, sha256 `b69f6dfa9937b250af21023a8127d74cf607af09b7447d830de2351be0c31738`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/3aea0c429934d790.bin` (4360132 B, sha256 `b69f6dfa9937b250af21023a8127d74cf607af09b7447d830de2351be0c31738`)
- `research/pupil_total_effects/results/student_mixture/s2z_auto/native/fit.rds.gz` (1282362 B, sha256 `ffcb76ad92b68de9d83c994a5df9755595620e3cfaade8e0cb59a40691be0042`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/2e7ddbc10c1ea6c6.bin` (1282362 B, sha256 `ffcb76ad92b68de9d83c994a5df9755595620e3cfaade8e0cb59a40691be0042`)
- `research/pupil_total_effects/results/student_mixture/s2z_auto/native/pupil-1.csv.gz` (4234522 B, sha256 `e1f48399c5acf53a87a0fd5ba5f3e9f79925e500e51436d3e8877d66518a7486`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/ed187367f42e13ae.bin` (4234522 B, sha256 `e1f48399c5acf53a87a0fd5ba5f3e9f79925e500e51436d3e8877d66518a7486`)
- `research/pupil_total_effects/results/student_mixture/s2z_auto/whmc/fit.jls.gz` (4987608 B, sha256 `e07fb62d519db83c6f5f724445e5ec9179c261fbf94329fe193c7fb9171f3cc6`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/0620e6f00197cc52.bin` (4987608 B, sha256 `e07fb62d519db83c6f5f724445e5ec9179c261fbf94329fe193c7fb9171f3cc6`)
- `research/pupil_total_effects/results/student_mixture/s2z_cp/native/fit.jls.gz` (4330190 B, sha256 `ba389f12ec728dcbd30a9fc0d442571a480c81b303e068ec873e9e3e9bc1af09`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/2d3e0983be9f64d0.bin` (4330190 B, sha256 `ba389f12ec728dcbd30a9fc0d442571a480c81b303e068ec873e9e3e9bc1af09`)
- `research/pupil_total_effects/results/student_mixture/s2z_cp/native/fit.rds.gz` (1270221 B, sha256 `0f5fe60e33dbfcac6247896369b02b94ed18ca258911c92dff4cf64a0185b15f`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/6cc8bf06795df86c.bin` (1270221 B, sha256 `0f5fe60e33dbfcac6247896369b02b94ed18ca258911c92dff4cf64a0185b15f`)
- `research/pupil_total_effects/results/student_mixture/s2z_cp/native/pupil-1.csv.gz` (4091621 B, sha256 `597f21c95f1411164ca0ceed554868529106ed74cb865722ca2495e29556fd28`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/59575ae43b3c08a8.bin` (4091621 B, sha256 `597f21c95f1411164ca0ceed554868529106ed74cb865722ca2495e29556fd28`)
- `research/pupil_total_effects/results/student_mixture/s2z_cp/whmc/fit.jls.gz` (4955015 B, sha256 `9b45f0b3d5b851e345d666b99066f298b92c5529981d1bd9639e39d218275046`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/3cc20af5bf95cf87.bin` (4955015 B, sha256 `9b45f0b3d5b851e345d666b99066f298b92c5529981d1bd9639e39d218275046`)
- `research/pupil_total_effects/results/student_mixture/s2z_ncp/native/fit.jls.gz` (4352161 B, sha256 `ee989fb1120245cb0eb58635249944f8a9adde95f494b234a3593e7967905a3f`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/b23c8e8d779c5404.bin` (4352161 B, sha256 `ee989fb1120245cb0eb58635249944f8a9adde95f494b234a3593e7967905a3f`)
- `research/pupil_total_effects/results/student_mixture/s2z_ncp/native/fit.rds.gz` (1260835 B, sha256 `854c2448fd84b4e646ac3b77b261af46ce496fc81bfdbc4cde897e1c90086e33`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/4ee894447014cc98.bin` (1260835 B, sha256 `854c2448fd84b4e646ac3b77b261af46ce496fc81bfdbc4cde897e1c90086e33`)
- `research/pupil_total_effects/results/student_mixture/s2z_ncp/native/pupil-1.csv.gz` (4099462 B, sha256 `a675c7f4b7f3b94ae066c667ba732de61475b4282f03c3e201861ca1fdb0f543`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/055a3dfed3101fce.bin` (4099462 B, sha256 `a675c7f4b7f3b94ae066c667ba732de61475b4282f03c3e201861ca1fdb0f543`)
- `research/pupil_total_effects/results/student_mixture/s2z_ncp/whmc/fit.jls.gz` (4979201 B, sha256 `77e625577813278c01042207ab2be6465f1445990fe7549a64bb4b7a32ae8057`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/530f07a50a3a8ecf.bin` (4979201 B, sha256 `77e625577813278c01042207ab2be6465f1445990fe7549a64bb4b7a32ae8057`)
- `research/radon_centering/results/ppc_intervals.tsv` (2070352 B, sha256 `d43446009113b8dd01de25809480c8f9a035f2c98a05ca5d078a984938b4760f`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/6e3576a18fcdb0c7.bin` (2070352 B, sha256 `d43446009113b8dd01de25809480c8f9a035f2c98a05ca5d078a984938b4760f`)
- `research/rbest_centering/results/AS/ordinary_pairs.aov.json` (2378142 B, sha256 `bb4301c253c57d476c3bf4b46d15179eb6f6dd33c9709c7d2be328e79ddfddb1`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/2142bbe0052d01d8.bin` (2378142 B, sha256 `bb4301c253c57d476c3bf4b46d15179eb6f6dd33c9709c7d2be328e79ddfddb1`)
- `research/rbest_centering/results/AS/ordinary_pairs.png` (1054863 B, sha256 `3eb39b479c3ffbf9d5a88dfc11e8ac5e1ffaef010836edb5e44a47f1e2201fd9`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/abaebd131756fc52.bin` (1054863 B, sha256 `3eb39b479c3ffbf9d5a88dfc11e8ac5e1ffaef010836edb5e44a47f1e2201fd9`)
- `research/rbest_centering/results/AS/ordinary_pairs.tsv` (6665836 B, sha256 `f0825f1c7b9b7fcaee74bb6ba16adcd06dba56365c6509d58bf6ffaf661023d2`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/a3d7ec5ff1ee6e3d.bin` (6665836 B, sha256 `f0825f1c7b9b7fcaee74bb6ba16adcd06dba56365c6509d58bf6ffaf661023d2`)
- `research/rbest_centering/results/AS/total_pairs.aov.json` (2372065 B, sha256 `b81b8dcb412abb09da738841ac27ac7650fa3c507428bb8fc80d5e5e9b786e36`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/611929c43911ab28.bin` (2372065 B, sha256 `b81b8dcb412abb09da738841ac27ac7650fa3c507428bb8fc80d5e5e9b786e36`)
- `research/rbest_centering/results/AS/total_pairs.png` (1303966 B, sha256 `00a312f9938faa034e401964f716c35dc5c746d578483fc29bdf5e43b13f7893`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/105291385acd145a.bin` (1303966 B, sha256 `00a312f9938faa034e401964f716c35dc5c746d578483fc29bdf5e43b13f7893`)
- `research/rbest_centering/results/AS/total_pairs.tsv` (6634755 B, sha256 `71b817673622589269312beb4e898193cd42c923190101d4940b8bbb46301758`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/a613f0bbb47dbbe5.bin` (6634755 B, sha256 `71b817673622589269312beb4e898193cd42c923190101d4940b8bbb46301758`)
- `research/rbest_centering/results/crohn/ordinary_pairs.aov.json` (2337446 B, sha256 `0f94447876ef2d4e7c7d7103e2f78595617b3b57e1211a4207a3635970572d1e`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/1d1da2b72aa12f24.bin` (2337446 B, sha256 `0f94447876ef2d4e7c7d7103e2f78595617b3b57e1211a4207a3635970572d1e`)
- `research/rbest_centering/results/crohn/ordinary_pairs.png` (1138001 B, sha256 `c57445b00260d83e6d56133cfa85f1416fd852491901857669238729f52e2a27`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/76faf586882d3709.bin` (1138001 B, sha256 `c57445b00260d83e6d56133cfa85f1416fd852491901857669238729f52e2a27`)
- `research/rbest_centering/results/crohn/ordinary_pairs.tsv` (6462964 B, sha256 `89c6f5ac1d07c3572c60aad5b24c9811604f2d7e82ea70798a4cf566233897a4`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/09ecc178175c526b.bin` (6462964 B, sha256 `89c6f5ac1d07c3572c60aad5b24c9811604f2d7e82ea70798a4cf566233897a4`)
- `research/rbest_centering/results/crohn/total_pairs.aov.json` (2347177 B, sha256 `69179a7c25d785d21873d441e773db51a450d7cb9d63976ff2d693d346916eed`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/0f76ee4c3a34b36d.bin` (2347177 B, sha256 `69179a7c25d785d21873d441e773db51a450d7cb9d63976ff2d693d346916eed`)
- `research/rbest_centering/results/crohn/total_pairs.png` (1189867 B, sha256 `7c5517710ea4e5bec31cbaa9281789f2e26b2a37041ce5ffa0cb166130a7e6ab`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/4b2058de6b3d13f9.bin` (1189867 B, sha256 `7c5517710ea4e5bec31cbaa9281789f2e26b2a37041ce5ffa0cb166130a7e6ab`)
- `research/rbest_centering/results/crohn/total_pairs.tsv` (6511391 B, sha256 `7bbf666573f37662ac3c1e4879a7fa7e7174cc5c7710352559e9c9f10f319179`)
  - part 0: `/home/niko/.local/state/kb-agents/uploads/ac8f28212e30e5a5.bin` (6511391 B, sha256 `7bbf666573f37662ac3c1e4879a7fa7e7174cc5c7710352559e9c9f10f319179`)
