# Source Materials

This simulator was generated from a DBbun LLC source concept titled:

**Reproducibility in Clinical AI Requires Modeling Measurement Error**

The source concept argues that clinical AI systems trained on electronic medical record (EMR) data often treat recorded values as biological ground truth, even though those values may be affected by pre-analytical laboratory artifacts, device calibration, transient physiological variation, and diagnostic threshold definitions.

The simulator operationalizes this idea through five synthetic scenarios:

1. False critical potassium from hemolyzed specimens
2. Borderline HbA1c label instability
3. Borderline systolic blood pressure and white-coat effect
4. Cross-site calibration drift
5. Diagnostic threshold sensitivity analysis

The `input/` folder includes the original prompt used to generate the simulator and a conceptual figure used to describe measurement uncertainty in EMR-derived clinical AI pipelines.

The unpublished source manuscript itself is not included in this public repository. This avoids making draft manuscript text public while preserving enough context for reviewers to understand the simulator's origin and purpose.
