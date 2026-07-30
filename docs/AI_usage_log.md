# AI Usage Log

Running record of how AI tools were used on LabRack Vision QA. Every output was
verified against the repository, tests, or a real run before we kept it, and the
final project reflects the team's own understanding and implementation. We use
Gemini as a learning collaborator — to reason through problems, debug, and explain
concepts — while the design decisions, code, and results are ours.

| Date | Tool | What we used it for | What we learned | What we kept or changed |
|------|------|---------------------|-----------------|-------------------------|
| 2026-07-25 | Gemini | Debug why pixel coordinates in a detection dictionary were not whole numbers | YOLO coordinates can stay as floating-point values until drawing or display; rounding too early can lose precision. | We printed the dictionary in a loop and kept conversion at the output boundary. |
| 2026-07-29 | Gemini | Review the repository with us against the runbook, architecture decisions, and final-build rubric | The project already had useful structure, so the safest path was to preserve working code and close specific evidence gaps. | We kept the reusable `src/` pipeline and notebook structure and made bounded fixes. |
| 2026-07-29 | Gemini | Plan how to incorporate the partial Roboflow export and create leakage-resistant splits | Adjacent images of one physical rack should not be scattered across train, validation, and test. | We kept the source export, remapped its three labels into the four-class schema, and used a rack-group split manifest. |
| 2026-07-29 | Gemini | Diagnose smoke-training failures and make training portable across CUDA, Apple MPS, and CPU | Ultralytics path resolution and MPS worker settings can fail before model quality is tested. | We corrected the dataset root, used `workers=0` for MPS, and kept preflight tests before full training. |
| 2026-07-29 | Gemini | Plan the baseline training run and package only real outputs | A saved best checkpoint and generated metrics are stronger evidence than terminal claims alone. | We kept the actual plots, tables, annotated images, JSON, and summary produced by the run. |
| 2026-07-29 | Gemini | Work through the YOLO11 model-size and resolution progression | Higher resolution helped empty-slot recall more than simply moving to YOLO11m on this small dataset. | We selected YOLO11s/960 at epoch 82 and rejected the weaker YOLO11m/640 candidate. |
| 2026-07-29 | Gemini | Plan how to rebuild the final evidence and deliverables from the selected checkpoint | Results, limitations, and a visible failure case all need to appear in the same final story. | We kept the measured 0.825 validation mAP50, 2.07-second CLI timing, real failure case, and human-review boundary in the notebook and slides. |

## Notes

- Planning and structure: used for organizing the blueprint and milestone plan.
- Code and debugging: AI suggestions were applied in small milestones and
  verified with tests or actual program runs.
- Explanation: class mapping, grouped splitting, relative dataset paths, and
  held-out evaluation were checked against the resulting files and metrics.
- All metrics, detections, and results are produced by real runs — never fabricated.
- Gemini is used as a learning collaborator; the technical decisions recorded here
  are ones the team can explain and defend during the demonstration.
