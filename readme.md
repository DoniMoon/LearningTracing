# Assessment-Learning Framework for Knowledge Tracing

## About this Repo

This is a temporary repository for anonymous review.
This repository builds upon two background works:

1. [JEDM 2020 - When is deep learning the best approach to knowledge tracing?]((https://github.com/theophilegervet/learner-performance-prediction))
2. [EDM 2025 - Using Large Multimodal Models to Extract Knowledge Components for Knowledge Tracing from Multimedia Question Information](https://github.com/DoniMoon/LLMKT)

The codebase added to [1] for measuring KT performance is documented in the [KT_Bench/readme.md](KT_Bench/readme.md) file within the `KT_Bench` directory.
For the research in [2], code was reused to extract content datasets from the [CMU DataShop](https://pslcdatashop.web.cmu.edu/index.jsp) datasets.

## RQ Based Table of Contents

Chapter 3 of this paper corresponds to RQ 1, and Chapter 4 corresponds to RQ 2.

### RQ 1. How much learning do existing KT methods capture?

#### RQ 1-1: How accurate can predictions be even when excluding learning?

#### RQ 1-2: Interpreting the predictions of each KT method by decomposing them into Assessment and Learning components.

### RQ 2. Is it reliable to rely on declarative knowledge? 

### RQ 2-1: How should KCs be grouped to demonstrate learning curve behavior?

### RQ 2-2: When KCs are modeled using Base-level Activation, will the correctness predictions exhibit a learning curve?

## Repo Structure
* **Datas** - Datasets used. Due to the licensing and file size issues of the OLI dataset, additional downloads are required from CMU DataShop.
* **KT_Bench** - Directory containing the code used for the verification of RQ 1.
* **Declarative_modeling** - Directory containing the code used for RQ 2.