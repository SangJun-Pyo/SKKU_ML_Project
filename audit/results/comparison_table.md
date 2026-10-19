| check | expected (key_numbers) | recomputed | tolerance | status |
|---|---|---|---|---|
| raw rows | 129971 | 129971 | 0 | PASS |
| after exact-dup (excl. Unnamed: 0) | 119988 | 119988 | 0 | PASS |
| after required fields | 111537 | 111537 | 0 | PASS |
| after dup normalized descriptions | 111510 | 111510 | 0 | PASS |
| clean row-id set identical to own cleaning | True | True | 0 | PASS |
| split rows train | 71381 | 71381 | 0 | PASS |
| unique titles train | 70748 | 70748 | 0 | PASS |
| split rows valid | 17843 | 17843 | 0 | PASS |
| unique titles valid | 17688 | 17688 | 0 | PASS |
| split rows test | 22286 | 22286 | 0 | PASS |
| unique titles test | 22109 | 22109 | 0 | PASS |
| title_norm overlap train-valid | 0 | 0 | 0 | PASS |
| title_norm overlap train-test | 0 | 0 | 0 | PASS |
| title_norm overlap valid-test | 0 | 0 | 0 | PASS |
| split reproduced from own cleaning (row mismatches) | 0 | 0 | 0 | PASS |
| Pearson r (raw price) | 0.4166 | 0.4166 | 0.0001 | PASS |
| Pearson r (log price) | 0.6178 | 0.6178 | 0.0001 | PASS |
| r^2 log price in-sample | 0.3816 | 0.3816 | 0.0001 | PASS |
| Spearman rho | 0.6126 | 0.6126 | 0.0001 | PASS |
| share >= 90 (%) | 37.7527 | 37.7527 | 0.001 | PASS |
| bucket <=$10 n | 5913 | 5913 | 0 | PASS |
| bucket <=$10 median / share>=90 | (85.0, 1.1838) | (85.0, 1.1838) | 0 | PASS |
| bucket ($10,$20] n | 36806 | 36806 | 0 | PASS |
| bucket ($10,$20] median / share>=90 | (87.0, 12.8756) | (87.0, 12.8756) | 0 | PASS |
| bucket ($20,$30] n | 24798 | 24798 | 0 | PASS |
| bucket ($20,$30] median / share>=90 | (88.0, 34.0229) | (88.0, 34.0229) | 0 | PASS |
| bucket ($30,$50] n | 25519 | 25519 | 0 | PASS |
| bucket ($30,$50] median / share>=90 | (90.0, 56.0563) | (90.0, 56.0563) | 0 | PASS |
| bucket ($50,$100] n | 15245 | 15245 | 0 | PASS |
| bucket ($50,$100] median / share>=90 | (91.0, 76.4185) | (91.0, 76.4185) | 0 | PASS |
| bucket >$100 n | 3229 | 3229 | 0 | PASS |
| bucket >$100 median / share>=90 | (93.0, 89.7182) | (93.0, 89.7182) | 0 | PASS |
| mean valid MAE (from saved preds) | 2.5142 | 2.5142 | 0.0001 | PASS |
| mean valid RMSE (from saved preds) | 3.0690 | 3.0690 | 0.0001 | PASS |
| mean valid R2 (from saved preds) | -0.0000 | -0.0000 | 0.0001 | PASS |
| mean test MAE (from saved preds) | 2.5336 | 2.5336 | 0.0001 | PASS |
| mean test RMSE (from saved preds) | 3.0943 | 3.0943 | 0.0001 | PASS |
| mean test R2 (from saved preds) | -0.0000 | -0.0000 | 0.0001 | PASS |
| ridge_price valid MAE (from saved preds) | 1.9288 | 1.9288 | 0.0001 | PASS |
| ridge_price valid RMSE (from saved preds) | 2.4179 | 2.4179 | 0.0001 | PASS |
| ridge_price valid R2 (from saved preds) | 0.3793 | 0.3793 | 0.0001 | PASS |
| ridge_price test MAE (from saved preds) | 1.9331 | 1.9331 | 0.0001 | PASS |
| ridge_price test RMSE (from saved preds) | 2.4345 | 2.4345 | 0.0001 | PASS |
| ridge_price test R2 (from saved preds) | 0.3810 | 0.3810 | 0.0001 | PASS |
| ridge_meta valid MAE (from saved preds) | 2.2408 | 2.2408 | 0.0001 | PASS |
| ridge_meta valid RMSE (from saved preds) | 2.7821 | 2.7821 | 0.0001 | PASS |
| ridge_meta valid R2 (from saved preds) | 0.1782 | 0.1782 | 0.0001 | PASS |
| ridge_meta test MAE (from saved preds) | 2.2539 | 2.2539 | 0.0001 | PASS |
| ridge_meta test RMSE (from saved preds) | 2.8013 | 2.8013 | 0.0001 | PASS |
| ridge_meta test R2 (from saved preds) | 0.1804 | 0.1804 | 0.0001 | PASS |
| ridge_meta_price valid MAE (from saved preds) | 1.8176 | 1.8176 | 0.0001 | PASS |
| ridge_meta_price valid RMSE (from saved preds) | 2.2840 | 2.2840 | 0.0001 | PASS |
| ridge_meta_price valid R2 (from saved preds) | 0.4462 | 0.4462 | 0.0001 | PASS |
| ridge_meta_price test MAE (from saved preds) | 1.8176 | 1.8176 | 0.0001 | PASS |
| ridge_meta_price test RMSE (from saved preds) | 2.2996 | 2.2996 | 0.0001 | PASS |
| ridge_meta_price test R2 (from saved preds) | 0.4477 | 0.4477 | 0.0001 | PASS |
| rf_meta_price valid MAE (from saved preds) | 1.7437 | 1.7437 | 0.0001 | PASS |
| rf_meta_price valid RMSE (from saved preds) | 2.2144 | 2.2144 | 0.0001 | PASS |
| rf_meta_price valid R2 (from saved preds) | 0.4794 | 0.4794 | 0.0001 | PASS |
| rf_meta_price test MAE (from saved preds) | 1.7398 | 1.7398 | 0.0001 | PASS |
| rf_meta_price test RMSE (from saved preds) | 2.2192 | 2.2192 | 0.0001 | PASS |
| rf_meta_price test R2 (from saved preds) | 0.4857 | 0.4857 | 0.0001 | PASS |
| xgb_meta_price valid MAE (from saved preds) | 1.7135 | 1.7135 | 0.0001 | PASS |
| xgb_meta_price valid RMSE (from saved preds) | 2.1781 | 2.1781 | 0.0001 | PASS |
| xgb_meta_price valid R2 (from saved preds) | 0.4963 | 0.4963 | 0.0001 | PASS |
| xgb_meta_price test MAE (from saved preds) | 1.7118 | 1.7118 | 0.0001 | PASS |
| xgb_meta_price test RMSE (from saved preds) | 2.1856 | 2.1856 | 0.0001 | PASS |
| xgb_meta_price test R2 (from saved preds) | 0.5011 | 0.5011 | 0.0001 | PASS |
| lasso_meta_price valid MAE (from saved preds) | 1.8178 | 1.8178 | 0.0001 | PASS |
| lasso_meta_price valid RMSE (from saved preds) | 2.2840 | 2.2840 | 0.0001 | PASS |
| lasso_meta_price valid R2 (from saved preds) | 0.4461 | 0.4461 | 0.0001 | PASS |
| lasso_meta_price test MAE (from saved preds) | 1.8178 | 1.8178 | 0.0001 | PASS |
| lasso_meta_price test RMSE (from saved preds) | 2.2996 | 2.2996 | 0.0001 | PASS |
| lasso_meta_price test R2 (from saved preds) | 0.4477 | 0.4477 | 0.0001 | PASS |
| enet_meta_price valid MAE (from saved preds) | 1.8182 | 1.8182 | 0.0001 | PASS |
| enet_meta_price valid RMSE (from saved preds) | 2.2846 | 2.2846 | 0.0001 | PASS |
| enet_meta_price valid R2 (from saved preds) | 0.4459 | 0.4459 | 0.0001 | PASS |
| enet_meta_price test MAE (from saved preds) | 1.8184 | 1.8184 | 0.0001 | PASS |
| enet_meta_price test RMSE (from saved preds) | 2.3004 | 2.3004 | 0.0001 | PASS |
| enet_meta_price test R2 (from saved preds) | 0.4473 | 0.4473 | 0.0001 | PASS |
| xgb_meta_price_winery valid MAE (from saved preds) | 1.6461 | 1.6461 | 0.0001 | PASS |
| xgb_meta_price_winery valid RMSE (from saved preds) | 2.0956 | 2.0956 | 0.0001 | PASS |
| xgb_meta_price_winery valid R2 (from saved preds) | 0.5337 | 0.5337 | 0.0001 | PASS |
| xgb_meta_price_winery test MAE (from saved preds) | 1.6307 | 1.6307 | 0.0001 | PASS |
| xgb_meta_price_winery test RMSE (from saved preds) | 2.0855 | 2.0855 | 0.0001 | PASS |
| xgb_meta_price_winery test R2 (from saved preds) | 0.5457 | 0.5457 | 0.0001 | PASS |
| text_only_ridge valid MAE (from saved preds) | 1.3425 | 1.3425 | 0.0001 | PASS |
| text_only_ridge valid RMSE (from saved preds) | 1.7057 | 1.7057 | 0.0001 | PASS |
| text_only_ridge valid R2 (from saved preds) | 0.6911 | 0.6911 | 0.0001 | PASS |
| text_only_ridge test MAE (from saved preds) | 1.3267 | 1.3267 | 0.0001 | PASS |
| text_only_ridge test RMSE (from saved preds) | 1.6838 | 1.6838 | 0.0001 | PASS |
| text_only_ridge test R2 (from saved preds) | 0.7039 | 0.7039 | 0.0001 | PASS |
| text_ridge valid MAE (from saved preds) | 1.1699 | 1.1699 | 0.0001 | PASS |
| text_ridge valid RMSE (from saved preds) | 1.4913 | 1.4913 | 0.0001 | PASS |
| text_ridge valid R2 (from saved preds) | 0.7639 | 0.7639 | 0.0001 | PASS |
| text_ridge test MAE (from saved preds) | 1.1673 | 1.1673 | 0.0001 | PASS |
| text_ridge test RMSE (from saved preds) | 1.4796 | 1.4796 | 0.0001 | PASS |
| text_ridge test R2 (from saved preds) | 0.7713 | 0.7713 | 0.0001 | PASS |
| ridge_price test MAE (own refit) | 1.9331 | 1.9331 | 0.005 | PASS |
| ridge_price test R2 (own refit) | 0.3810 | 0.3810 | 0.005 | PASS |
| ridge_meta test MAE (own refit) | 2.2539 | 2.2539 | 0.005 | PASS |
| ridge_meta test R2 (own refit) | 0.1804 | 0.1804 | 0.005 | PASS |
| ridge_meta_price test MAE (own refit) | 1.8176 | 1.8176 | 0.005 | PASS |
| ridge_meta_price test R2 (own refit) | 0.4477 | 0.4477 | 0.005 | PASS |
| xgb_meta_price test MAE (own refit, seed 7) | 1.7118 | 1.7130 | 0.01 | PASS |
| xgb_meta_price test R2 (own refit, seed 7) | 0.5011 | 0.5008 | 0.01 | PASS |
| test MAE gap ridge - main | 0.1058 | 0.1058 | 0.0001 | PASS |
| test MAE gap CI low (own seed) | 0.0976 | 0.0969 | 0.005 | PASS |
| test MAE gap CI high (own seed) | 0.1143 | 0.1143 | 0.005 | PASS |
| sim $20 Random mean points | 86.5140 | 86.5140 | 0.001 | PASS |
| sim $20 Random share>=90 % | 11.0950 | 11.0950 | 0.001 | PASS |
| sim $20 Random mean price | 15.0726 | 15.0727 | 0.001 | PASS |
| sim $20 MaxPrice mean points | 87.5790 | 87.5790 | 0.001 | PASS |
| sim $20 MaxPrice share>=90 % | 22.0000 | 22.0000 | 0.001 | PASS |
| sim $20 MaxPrice mean price | 19.9490 | 19.9490 | 0.001 | PASS |
| sim $20 MaxPrice share>=90 % (tie-EV re-pick) | 22.0000 | 21.8816 | 1.0 | PASS |
| sim $20 Ridge mean points | 88.6480 | 88.6480 | 0.001 | PASS |
| sim $20 Ridge share>=90 % | 37.2000 | 37.2000 | 0.001 | PASS |
| sim $20 Ridge mean price | 18.2260 | 18.2260 | 0.001 | PASS |
| sim $20 Ridge share>=90 % (tie-EV re-pick) | 37.2000 | 37.2000 | 1.0 | PASS |
| sim $20 Main mean points | 88.8990 | 88.8990 | 0.001 | PASS |
| sim $20 Main share>=90 % | 41.9000 | 41.9000 | 0.001 | PASS |
| sim $20 Main mean price | 18.2360 | 18.2360 | 0.001 | PASS |
| sim $20 Main share>=90 % (tie-EV re-pick) | 41.9000 | 41.9000 | 1.0 | PASS |
| sim $20 all candidates within budget | True | True | 0 | PASS |
| sim $20 every market 20 distinct titles | True | True | 0 | PASS |
| sim $20 MaxPrice saved pick is a valid argmax (of 1000) | 1000 | 1000 | 0 | PASS |
| sim $20 Ridge saved pick is a valid argmax (of 1000) | 1000 | 1000 | 0 | PASS |
| sim $20 Main saved pick is a valid argmax (of 1000) | 1000 | 1000 | 0 | PASS |
| sim $30 Random mean points | 87.1553 | 87.1553 | 0.001 | PASS |
| sim $30 Random share>=90 % | 19.6450 | 19.6450 | 0.001 | PASS |
| sim $30 Random mean price | 18.9980 | 18.9980 | 0.001 | PASS |
| sim $30 MaxPrice mean points | 88.6820 | 88.6820 | 0.001 | PASS |
| sim $30 MaxPrice share>=90 % | 41.0000 | 41.0000 | 0.001 | PASS |
| sim $30 MaxPrice mean price | 29.3880 | 29.3880 | 0.001 | PASS |
| sim $30 MaxPrice share>=90 % (tie-EV re-pick) | 41.0000 | 41.4750 | 1.0 | PASS |
| sim $30 Ridge mean points | 89.7720 | 89.7720 | 0.001 | PASS |
| sim $30 Ridge share>=90 % | 60.5000 | 60.5000 | 0.001 | PASS |
| sim $30 Ridge mean price | 25.5050 | 25.5050 | 0.001 | PASS |
| sim $30 Ridge share>=90 % (tie-EV re-pick) | 60.5000 | 60.5000 | 1.0 | PASS |
| sim $30 Main mean points | 89.9800 | 89.9800 | 0.001 | PASS |
| sim $30 Main share>=90 % | 64.7000 | 64.7000 | 0.001 | PASS |
| sim $30 Main mean price | 25.1420 | 25.1420 | 0.001 | PASS |
| sim $30 Main share>=90 % (tie-EV re-pick) | 64.7000 | 64.7000 | 1.0 | PASS |
| sim $30 all candidates within budget | True | True | 0 | PASS |
| sim $30 every market 20 distinct titles | True | True | 0 | PASS |
| sim $30 MaxPrice saved pick is a valid argmax (of 1000) | 1000 | 1000 | 0 | PASS |
| sim $30 Ridge saved pick is a valid argmax (of 1000) | 1000 | 1000 | 0 | PASS |
| sim $30 Main saved pick is a valid argmax (of 1000) | 1000 | 1000 | 0 | PASS |
| sim $50 Random mean points | 87.8334 | 87.8334 | 0.001 | PASS |
| sim $50 Random share>=90 % | 29.8500 | 29.8500 | 0.001 | PASS |
| sim $50 Random mean price | 24.9464 | 24.9464 | 0.001 | PASS |
| sim $50 MaxPrice mean points | 90.1950 | 90.1950 | 0.001 | PASS |
| sim $50 MaxPrice share>=90 % | 63.6000 | 63.6000 | 0.001 | PASS |
| sim $50 MaxPrice mean price | 47.4060 | 47.4060 | 0.001 | PASS |
| sim $50 MaxPrice share>=90 % (tie-EV re-pick) | 63.6000 | 63.5350 | 1.0 | PASS |
| sim $50 Ridge mean points | 90.9670 | 90.9670 | 0.001 | PASS |
| sim $50 Ridge share>=90 % | 77.5000 | 77.5000 | 0.001 | PASS |
| sim $50 Ridge mean price | 41.7390 | 41.7390 | 0.001 | PASS |
| sim $50 Ridge share>=90 % (tie-EV re-pick) | 77.5000 | 77.5000 | 1.0 | PASS |
| sim $50 Main mean points | 91.1020 | 91.1020 | 0.001 | PASS |
| sim $50 Main share>=90 % | 80.7000 | 80.7000 | 0.001 | PASS |
| sim $50 Main mean price | 41.4610 | 41.4610 | 0.001 | PASS |
| sim $50 Main share>=90 % (tie-EV re-pick) | 80.7000 | 80.7500 | 1.0 | PASS |
| sim $50 all candidates within budget | True | True | 0 | PASS |
| sim $50 every market 20 distinct titles | True | True | 0 | PASS |
| sim $50 MaxPrice saved pick is a valid argmax (of 1000) | 1000 | 1000 | 0 | PASS |
| sim $50 Ridge saved pick is a valid argmax (of 1000) | 1000 | 1000 | 0 | PASS |
| sim $50 Main saved pick is a valid argmax (of 1000) | 1000 | 1000 | 0 | PASS |
| $30 Main_minus_MaxPrice__ge90_pct_points diff | 23.7000 | 23.7000 | 0.001 | PASS |
| $30 Main_minus_MaxPrice__ge90_pct_points CI low (own seed) | 20.0000 | 20.1000 | 0.5 | PASS |
| $30 Main_minus_MaxPrice__ge90_pct_points CI high (own seed) | 27.5000 | 27.3000 | 0.5 | PASS |
| $30 Ridge_minus_MaxPrice__ge90_pct_points diff | 19.5000 | 19.5000 | 0.001 | PASS |
| $30 Ridge_minus_MaxPrice__ge90_pct_points CI low (own seed) | 15.8000 | 15.8000 | 0.5 | PASS |
| $30 Ridge_minus_MaxPrice__ge90_pct_points CI high (own seed) | 23.2000 | 23.2000 | 0.5 | PASS |
| $30 Main_minus_MaxPrice__points_points diff | 1.2980 | 1.2980 | 0.001 | PASS |
| $30 Main_minus_MaxPrice__points_points CI low (own seed) | 1.1270 | 1.1230 | 0.03 | PASS |
| $30 Main_minus_MaxPrice__points_points CI high (own seed) | 1.4760 | 1.4770 | 0.03 | PASS |
| $30 Ridge_minus_MaxPrice__points_points diff | 1.0900 | 1.0900 | 0.001 | PASS |
| $30 Ridge_minus_MaxPrice__points_points CI low (own seed) | 0.9100 | 0.9090 | 0.03 | PASS |
| $30 Ridge_minus_MaxPrice__points_points CI high (own seed) | 1.2750 | 1.2740 | 0.03 | PASS |
| $30 Main_minus_MaxPrice__price_usd diff | -4.2460 | -4.2460 | 0.001 | PASS |
| $30 Main_minus_MaxPrice__price_usd CI low (own seed) | -4.5100 | -4.5110 | 0.03 | PASS |
| $30 Main_minus_MaxPrice__price_usd CI high (own seed) | -3.9850 | -3.9810 | 0.03 | PASS |
| points per price doubling | 2.0117 | 2.0117 | 0.001 | PASS |
| value map: n countries shown (n>=100) | 12 | 12 | 0 | PASS |
| value map: n varieties shown (n>=100) | 36 | 36 | 0 | PASS |
| value country Austria mean residual | 1.8162 | 1.8162 | 0.001 | PASS |
| value country Germany mean residual | 1.2634 | 1.2634 | 0.001 | PASS |
| value country Portugal mean residual | 0.9709 | 0.9709 | 0.001 | PASS |
| value country Australia mean residual | 0.3686 | 0.3686 | 0.001 | PASS |
| value country New Zealand mean residual | 0.2435 | 0.2435 | 0.001 | PASS |
| value country Argentina mean residual | -0.7173 | -0.7173 | 0.001 | PASS |
| value country Chile mean residual | -0.5676 | -0.5676 | 0.001 | PASS |
| value country Spain mean residual | -0.3579 | -0.3579 | 0.001 | PASS |
| value country US mean residual | -0.1639 | -0.1639 | 0.001 | PASS |
| value country Italy mean residual | -0.1075 | -0.1075 | 0.001 | PASS |
| value variety Grüner Veltliner mean residual | 2.0704 | 2.0704 | 0.001 | PASS |
| value variety Portuguese Red mean residual | 1.4459 | 1.4459 | 0.001 | PASS |
| value variety Riesling mean residual | 1.3478 | 1.3478 | 0.001 | PASS |
| value variety Pinot Gris mean residual | 0.8220 | 0.8220 | 0.001 | PASS |
| value variety Gewürztraminer mean residual | 0.6795 | 0.6795 | 0.001 | PASS |
| value variety Champagne Blend mean residual | -0.8684 | -0.8684 | 0.001 | PASS |
| value variety Merlot mean residual | -0.7609 | -0.7609 | 0.001 | PASS |
| value variety Cabernet Franc mean residual | -0.6873 | -0.6873 | 0.001 | PASS |
| value variety Sangiovese mean residual | -0.6472 | -0.6472 | 0.001 | PASS |
| value variety Pinot Grigio mean residual | -0.6224 | -0.6224 | 0.001 | PASS |
| critic Austria within-critic premium (builder def.) | 0.2245 | 0.1375 | 0.001 | FAIL |
| critic Austria premium, 'vs OTHER countries' def. (as labelled) | 0.2245 | 0.2245 | 0.05 | PASS |
| critic Germany within-critic premium (builder def.) | 1.4287 | 0.7712 | 0.001 | FAIL |
| critic Germany premium, 'vs OTHER countries' def. (as labelled) | 1.4287 | 1.4287 | 0.05 | PASS |
| critic Portugal within-critic premium (builder def.) | 0.8829 | 0.6652 | 0.001 | FAIL |
| critic Portugal premium, 'vs OTHER countries' def. (as labelled) | 0.8829 | 0.8829 | 0.05 | PASS |
| critic Argentina within-critic premium (builder def.) | -0.2816 | -0.2048 | 0.001 | FAIL |
| critic Argentina premium, 'vs OTHER countries' def. (as labelled) | -0.2816 | -0.2816 | 0.05 | PASS |
| critic Chile within-critic premium (builder def.) | -0.0815 | -0.0568 | 0.001 | FAIL |
| critic Chile premium, 'vs OTHER countries' def. (as labelled) | -0.0815 | -0.0815 | 0.05 | PASS |
| under-$20 list reproduced by pred-only ranking | True | True | 0 | PASS |
| under-$20 top10 mean actual | 91.2000 | 91.2000 | 0.001 | PASS |
| under-$20 top10 n>=90 | 10 | 10 | 0 | PASS |
| under-$20 pool share>=90 % | 11.0184 | 11.0184 | 0.001 | PASS |
| under-$20 Austrian picks (HARNESS_PLAN v1.1 says 7/10) | 7 | 8 | 0 | WARN |
| SHAP share log_price (saved sample) | 57.0638 | 57.0638 | 0.001 | PASS |
| SHAP share log_price (own refit, own sample) | 57.0638 | 56.8310 | 1.5 | PASS |
| SHAP share vintage (saved sample) | 14.6557 | 14.6557 | 0.001 | PASS |
| SHAP share vintage (own refit, own sample) | 14.6557 | 14.4384 | 1.5 | PASS |
| SHAP share variety (saved sample) | 10.9465 | 10.9465 | 0.001 | PASS |
| SHAP share variety (own refit, own sample) | 10.9465 | 10.7841 | 1.5 | PASS |
| SHAP share province (saved sample) | 9.7552 | 9.7552 | 0.001 | PASS |
| SHAP share province (own refit, own sample) | 9.7552 | 9.6235 | 1.5 | PASS |
| SHAP share country (saved sample) | 7.5788 | 7.5788 | 0.001 | PASS |
| SHAP share country (own refit, own sample) | 7.5788 | 8.3229 | 1.5 | PASS |
| text model test R2 (meta+price+TF-IDF) | 0.7713 | 0.7713 | 0.0001 | PASS |
