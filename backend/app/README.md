# Backend app layout

`main.py` creates the FastAPI app and registers three route groups:

| Module | Used by |
| --- | --- |
| `user_routes.py` | Main page and combined results page: model views, predictions, user edits, and aggregated results |
| `admin_routes.py` | Superadmin menu: datasets, training, import/export, chart settings, users, and maintenance |
| `export_compare_routes.py` | Exported model comparison page: artifact preparation, overlap check, and preview |

`api_common.py` holds request parsing and authorization helpers shared by the
route groups. Routes keep their existing `/api/...` paths.

`ml_service.py` owns the active model and dataset state. Its focused mixins
implement dataset provenance, training, charts, artifact import/export,
predictions, and interactive edits. `model_compare_prepare.py`,
`model_compare_metrics.py`, and `model_compare_preview.py` contain comparison
of exported artifacts. `ml_shared.py` holds artifact constants and the runtime
used to predict from an imported artifact.

The exported comparison code reads prepared artifacts supplied with each
request. The main-training performance option also reads the active model's
training data through `MLService`.
