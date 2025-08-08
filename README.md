# MixToSmell

# Dilution-aware primary odor maps for single and multi-component odor mixture ratings

Ella Rajaonson^1^, Gary Tom^1^, Rana Barghout^1^, Michelle Collins^1^, Jia Meng Wang^1^, Benjamin Sanchez-Lengeling^1^,^2^
^1^ University of Toronto
^2^ Vector Institute

## Summary Sentence
Our hierarchical approach predicts high-dimensional RATA odor profiles by first training a dilution-aware GNN with a multi-component structural loss, then using its learned embeddings with XGBoost to model complex mixtures.

## Background/Introduction
The prediction of odor perception from molecular structure is a significant challenge in sensory neuroscience, largely underexplored compared to other sensory modalities. While recent advances like the Principal Odor Map (POM) have established the utility of Graph Neural Networks (GNNs) for learning single-molecule representations [1], real-world olfaction involves complexities of dilution and multi-component mixtures. These phenomena introduce non-linear perceptual effects that are not captured by simple linear models.
Our motivation is to extend the representational capacity of GNNs to these more complex and realistic scenarios. We hypothesized that the perceptual effect of dilution could be modeled by conditioning a molecular embedding, and that dilution-aware embeddings could serve as input features for a low-data mixture prediction task. Our approach introduces two novelties: 1) a context-conditioning mechanism using a FiLM layer to create a Dilution-Aware POM (D-POM), and 2) a custom, multi-component loss function designed to capture both the magnitude and the high-dimensional structure of the sparse RATA perceptual vectors.

## Methods
${image?fileName=dream2025%5Foverview%2Epng&align=Center&scale=100&responsive=true&altText=}

Our methodology is a two-stage, hierarchical approach designed to address the specific constraints and objectives of each challenge task. The overall approach first develops a context-aware molecular representation for single molecules (Task 1) and then leverages these representations in a separate, data-efficient model for mixtures (Task 2). All models were implemented using PyTorch, PyTorch Geometric, and Scikit-learn.

**Task 1: Dilution-Aware POM (D-POM) for Single Molecules**

1.  **Molecular Representation:** We built upon the POM architecture [1,2,3,4,5], using a GraphNets-based GNN [7] to learn molecular embeddings. Input molecules, represented as SMILES strings, were canonicalized and converted into graph structures. Each graph consists of nodes (atoms) with 85-dimensional feature vectors (atomic number, charge, hybridization, etc.) and edges (bonds) with 14-dimensional feature vectors (bond type, conjugation, etc.). The GNN processes this graph to produce a final n-dimensional embedding vector for the molecule.

2.  **Dilution Conditioning:** To make the model dilution-aware, we conditioned the final molecular embedding using a Feature-wise Linear Modulation (FiLM) layer [9]. The normalized dilution value is passed through a small MLP to generate a scaling (`γ`) and shifting (`β`) vector, which then modulates the molecular embedding. This allows the model to learn a non-linear transformation of the odor representation based on dilution.

3.  **Prediction and Loss Function:** The conditioned embedding is passed to a 3-layer MLP which outputs the 51-dimensional RATA score vector. Given the sparse, non-Gaussian nature of the RATA labels, we designed a custom multi-component loss function to enforce structural correctness on the predicted vector:
    *   **Sparsity (BCE Loss):** A binary cross-entropy term on binarized labels (0 if rating is 0, 1 otherwise) to explicitly teach the model when values are non-zero.
    *   **Magnitude (RMSE Loss):** A standard root-mean-squared error term to capture the intensity of the non-zero ratings.
    *   **Vector Structure (Pearson & Cosine):** To ensure the relative pattern of the RATA vector is correct, we included two structural loss terms: `1 - Pearson correlation` and `1 - Cosine similarity` between the predicted and true vectors.
 The final loss is a weighted sum of these four components.

**Task 2: Mixture Prediction via Aggregated Embeddings**

1.  **Feature Generation:** Addressing the low-data regime of the mixture task, we adopted a transfer learning approach. We first used our best-trained D-POM model from Task 1 as a fixed feature extractor. For each component molecule in a mixture, we generated its dilution-aware embedding using the D-POM, feeding the model both its structure and its mole fraction within the mixture.

2.  **Mixture Representation and Feature Selection:** For each mixture, we aggregated the set of its component embeddings into a single, fixed-size feature vector. This was achieved by computing element-wise statistics (mean, standard deviation, min, max) across the set of embeddings and concatenating them. This creates a permutation-invariant representation suitable for a tree-based model. We then performed feature selection using the mutual information criterion to retain only the most informative features from this aggregated vector.

3.  **Data Augmentation and Modeling:** To maximize the use of available data, we augmented the Task 2 training set with all data from Task 1, treating each single-molecule sample as a "mixture" of one component. The final RATA score predictions were made using an XGBoost regressor [8] trained on the selected, aggregated features. This was chosen for its robustness against overfitting in low-data scenarios and its ability to handle heterogeneous feature sets.


## Conclusion/Discussion

This work introduces a pragmatic, two-stage approach for predicting complex olfactory percepts. Our primary contribution is the development of a Dilution-Aware POM (D-POM) which leverages a FiLM layer for context conditioning and a novel, multi-component loss function to model the high-dimensional, sparse structure of RATA data.

The insight behind our approach was that by decoupling the problem, we could build a GNN representation on the richer single-molecule dataset (Task 1) and then redeploy these embeddings as features for the data-scarce mixture task (Task 2). This transfer learning approach, combined with XGBoost, was a strategy for navigating the low-data regime of mixture prediction. A limitation is that this two-stage approach is not end-to-end differentiable, precluding joint optimization.

Future work could focus on developing end-to-end mixture models, perhaps by incorporating attention mechanisms as explored in the POMMIX paper, once larger mixture datasets become available. Furthermore, exploring alternative aggregation and feature selection techniques could improve the performance of the mixture prediction model.

## References

1. Lee, B. K., Mayhew, E. J., Sanchez-Lengeling, B., Wei, J. N., Qian, W. W., Little, K. A., Andres, M., Nguyen, B. B., Moloy, T., Yasonik, J., Parker, J. K., Gerkin, R. C., Mainland, J. D., & Wiltschko, A. B. (2023). A principal odor map unifies diverse tasks in olfactory perception. Science, 381(6661), 999–1006. https://doi.org/10.1126/science.ade4401
2. Tom, G., Ser, C.T., Rajaonson, E.M., Lo, S., Park, H.S., Lee, B.K., Sanchez-Lengeling, B. (2025). From Molecules to Mixtures: Learning Representations of Olfactory Mixture Similarity using Inductive Biases. arXiv preprint arXiv:2501.16271.
3. Rajaonson, E.M., Kochi, M.R., Mendoza, L.M.M., Moosavi, S.M., Sanchez-Lengeling, B. (2025). CheMixHub: Datasets and Benchmarks for Chemical Mixture Property Prediction. arXiv preprint arXiv:2506.12231.
4. Barsainyan, A. A., Kumar, R., Saha, P., & Schmuker, M. (2023). OpenPOM - Open Principal Odor Map.
5. Keller, A., Gerkin, R. C., Guan, Y., Dhurandhar, A., Turu, G., Szalai, B., Mainland, J. D., Ihara, Y., Yu, C. W., Wolfinger, R., Vens, C., schietgat, leander, De Grave, K., Norel, R., Stolovitzky, G., Cecchi, G. A., Vosshall, L. B., & meyer, pablo. (2017). Predicting human olfactory perception from chemical features of odor molecules. Science, 355(6327), 820–826. https://doi.org/10.1126/science.aal2014
6. DREAM Olfactory Mixtures Prediction Challenge 2025 (syn64743570)
7. Battaglia, P.W., Hamrick, J.B., Bapst, V., Sanchez-Gonzalez, A., Zambaldi, V., Malinowski, M., Tacchetti, A., Raposo, D., Santoro, A., Faulkner, R., Gulcehre, C., Song, F., Ballard, A., Gilmer, J., Dahl, G., Vaswani, A., Allen, K., Nash, C., Langston, V., Dyer, C., Heess, N., Wierstra, D., Kohli, P., Botvinick, M., Vinyals, O., Li, Y., Pascanu, R. (2018). Relational inductive biases, deep learning, and graph networks. arXiv preprint arXiv:1806.01261.
8. Chen, T., Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. arXiv preprint arXiv:1603.02754.
9. Perez, E., Strub, F., Vries, H.D., Dumoulin, V., Courville, A. (2017). FiLM: Visual Reasoning with a General Conditioning Layer. arXiv preprint arXiv:1709.07871.

##Authors Statement
Please list all author’s contributions

