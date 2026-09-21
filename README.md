## 2026 Update: LV Mesh Motion Analysis

This year's material extends the cardiac motion analysis framework with
left-ventricular (LV) mesh-based motion analysis.

In addition to image-based cardiac motion estimation, we use time-resolved
LV meshes to represent and analyse the motion of the ventricular anatomy
throughout the cardiac cycle.

### LV Mesh Motion Pipeline

Cine Cardiac MRI
        ↓
LV Segmentation
        ↓
LV Mesh Generation
        ↓
Temporal Mesh Correspondence
        ↓
LV Motion Analysis
        ↓
Motion Visualisation and Quantification

The LV mesh at each cardiac phase provides a geometric representation of
the ventricular anatomy. By tracking corresponding mesh vertices across
the cardiac cycle, cardiac motion can be represented as vertex-wise
displacement trajectories.

For a mesh vertex \(v_i\), displacement relative to the reference
end-diastolic (ED) configuration can be expressed as:

\[
d_i(t) = v_i(t) - v_i(ED)
\]

This enables analysis and visualisation of regional LV motion throughout
the cardiac cycle.

### What we analyse

The mesh-based analysis can be used to investigate:

- LV motion from end-diastole (ED) to end-systole (ES)
- vertex-wise displacement
- regional patterns of contraction and relaxation
- temporal trajectories across the cardiac cycle
- 3D visualisation of ventricular motion
- subject-level comparison of cardiac motion

### Why use meshes?

Image registration provides dense voxel-level motion information, while
cardiac meshes provide a compact anatomical representation of ventricular
shape and motion. Combining these representations makes it possible to
study cardiac dynamics directly on the ventricular surface and facilitates
regional quantitative analysis.

### Practical Exercise

The practical component demonstrates how to:

1. Load a time-resolved LV mesh sequence.
2. Visualise the LV geometry at different cardiac phases.
3. identify the ED and ES configurations.
4. Calculate vertex-wise displacement from ED.
5. Visualise regional LV displacement on the 3D mesh.
6. Plot motion trajectories throughout the cardiac cycle.


# Run the code

| File                            | Description                                |
| ------------------------------- | ------------------------------------------ |
| `Registration_prediction.ipynb` | Main notebook for running prediction       |
| ` prediction/trained_weights/`  | Folder containing pretrained model weights |
| `Val_data/test_sample/`         | Small sample dataset for testing           |
# Download dataset link for training and validation
https://mega.nz/file/NHMSkC4R#9yvIKvQV9r9QBQ1RMN9kSSVx3ZYr6mN8v5pTy_n6gME

