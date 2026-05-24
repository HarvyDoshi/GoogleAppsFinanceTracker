# Project Structure

```text
trace/
├── .env.example
├── .gitignore
├── .ledger_id
├── 2.0
├── Dockerfile
├── README.md
├── __init__.py
├── all_financial_dashboard.html
├── all_financial_response.json
├── all_financial_transactions.csv
├── all_financial_transactions.json
├── client.py
├── configs
│   ├── env_config.yaml
│   └── grpo_config.yaml
├── credentials.json
├── credentials.json.json
├── data
│   ├── sft_demos.jsonl
│   └── sft_demos_v2.jsonl
├── docs
│   └── blog_post.md
├── environments
│   ├── __init__.py
│   └── trace_env
│       ├── __init__.py
│       ├── agents
│       │   ├── __init__.py
│       │   ├── memory.py
│       │   ├── planner.py
│       │   ├── retriever.py
│       │   └── verifier.py
│       ├── app.py
│       ├── core
│       │   ├── __init__.py
│       │   ├── env.py
│       │   ├── schemas.py
│       │   └── world_model.py
│       ├── rewards
│       │   ├── __init__.py
│       │   ├── anti_hack.py
│       │   └── reward_fn.py
│       └── tools
│           ├── __init__.py
│           ├── dashboard_renderer.py
│           ├── doc_tool.py
│           ├── drive_tool.py
│           ├── gmail_tool.py
│           ├── image_tool.py
│           ├── report_tool.py
│           ├── run_rapido_flow.py
│           ├── sheets_tool.py
│           └── transaction_parser.py
├── financial_report.docx
├── financial_report_1777157050.docx
├── financial_report_1777157984.docx
├── financial_report_1777162828.docx
├── generate_secrets.py
├── hf_secrets.txt
├── inference.py
├── models.py
├── notebooks
│   └── Trace_SFT_Training_Colab.ipynb
├── openenv.yaml
├── project_structure.md
├── pyproject.toml
├── rapido_dashboard.html
├── rapido_response.json
├── requirements.txt
├── run_all_financial_gmail.py
├── scratch
│   ├── add_test_row.py
│   ├── check_all_tabs.py
│   ├── check_correct_id.py
│   ├── check_grid.py
│   ├── check_id_3.py
│   ├── check_last_edit.py
│   ├── check_metadata.py
│   ├── check_owner.py
│   ├── check_service.py
│   ├── check_tabs.py
│   ├── check_token.py
│   ├── check_values.py
│   ├── debug_sheets.py
│   ├── find_sheet.py
│   ├── generate_tree.py
│   ├── get_email.py
│   ├── list_all_sheets.py
│   ├── read_large.py
│   ├── search_zomato.py
│   ├── test_broad.py
│   ├── test_direct.py
│   ├── test_full.py
│   ├── test_importlib.py
│   └── test_shim.py
├── scripts
│   ├── __init__.py
│   ├── evaluate.py
│   ├── wandb_reward_curve_demo.py
│   └── wandb_sft_dataset_analyzer.py
├── server
│   ├── __init__.py
│   ├── app.py
│   ├── requirements.txt
│   └── trace_environment.py
├── static
│   └── index.html
├── test_api.py
├── test_tools.py
├── token_drive.pkl
├── token_gmail.pkl
├── token_sheets.pkl
├── tools
├── training
│   ├── __init__.py
│   ├── callbacks.py
│   ├── dataset.py
│   ├── export_model.py
│   ├── generate_sft_data.py
│   ├── train_grpo.py
│   └── train_sft.py
├── uber_dashboard.html
└── uv.lock
```
