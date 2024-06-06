# Experiments

Available metrics:
- set:
  - trainEpoch
  - valEpoch
  - testEpoch
- metric names:
  - avg_edge_index_bg_mcc  (retrain walle at edge level)
  - avg_edge_index_bg_tn
  - avg_edge_index_bg_fp
  - avg_edge_index_bg_fn
  - avg_edge_index_bg_tp
  - avg_epi_node_auprc (retrain walle at node level)
  - avg_epi_node_mcc
  - avg_epi_node_tn
  - avg_epi_node_fp
  - avg_epi_node_fn
  - avg_epi_node_tp

- edgeLevel-onlyRecLoss.yaml
  - use rec loss only