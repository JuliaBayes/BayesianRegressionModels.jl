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
