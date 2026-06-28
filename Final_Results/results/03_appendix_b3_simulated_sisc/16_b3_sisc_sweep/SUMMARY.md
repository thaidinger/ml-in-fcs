# Appendix B.3 SISC Sweep Summary

Timed exploration completed 18 successful multi-pattern SISC runs against the shipped toy data
`fts-diffusion-ref/data/data_toy_l10-20_*`.

Paper targets for the multi-pattern experiment:

| Metric | Paper |
|---|---:|
| Average per-unit DTW | 0.01 |
| Jaccard / IoU | 0.784 |

Best observed runs:

| Selection | Seed | l_max | max_iters | Avg per-unit DTW | Boundary Jaccard tol=2 | Author-style interval IoU |
|---|---:|---:|---:|---:|---:|---:|
| Best DTW | 4 | 20 | 10 | 0.0321407309 | 0.2064741907 | 0.5948493619 |
| Best author-style interval IoU | 2 | 21 | 20 | 0.0381847482 | 0.3589990375 | 0.6417796781 |
| Best boundary Jaccard tol=2 | 0 | 20 | 20 | 0.0412214694 | 0.3608445298 | 0.6382161252 |

Group means:

| l_max | max_iters | Mean avg per-unit DTW | Mean boundary Jaccard tol=2 | Mean author-style interval IoU |
|---:|---:|---:|---:|---:|
| 20 | 10 | 0.036278 | 0.292211 | 0.619901 |
| 20 | 20 | 0.038716 | 0.319552 | 0.623487 |
| 21 | 20 | 0.037922 | 0.304105 | 0.616273 |

Conclusion: seed changes, increasing iterations from 10 to 20, and switching from `l_max=20` to `l_max=21`
did not reproduce the Appendix B.3 reported metrics. The reference-code interval IoU definition narrows the
segmentation gap compared with strict boundary matching, but it still tops out at 0.6418 versus the paper's
0.784. The centroid metric remains at least about 3.2x the reported 0.01 in the best run.
