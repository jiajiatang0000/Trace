from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml import command
from azure.ai.ml import Input
from azure.ai.ml.entities import Environment


# authenticate
credential = DefaultAzureCredential()

SUBSCRIPTION = "437ce2b6-c1d8-4df6-b067-fc9209c568e9"
RESOURCE_GROUP = "Multi-policy"
WS_NAME = "multi-policy-training"
# Get a handle to the workspace
ml_client = MLClient(
    credential=credential,
    subscription_id=SUBSCRIPTION,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WS_NAME,
)

# Verify that the handle works correctly.
# If you ge an error here, modify your SUBSCRIPTION, RESOURCE_GROUP, and WS_NAME in the previous cell.
ws = ml_client.workspaces.get(WS_NAME)
print(ws.location, ":", ws.resource_group)

# custom environment
custom_env_name = "custom-environment"

custom_job_env = Environment(
    name=custom_env_name,
    description="Custom environment for multi-policy training",
    conda_file="environment_0311.yaml",
    image="mcr.microsoft.com/azureml/minimal-ubuntu22.04-py39-cuda11.8-gpu-inference:latest",
)
custom_job_env = ml_client.environments.create_or_update(custom_job_env)

print(
    f"Environment with name {custom_job_env.name} is registered to workspace, the environment version is {custom_job_env.version}"
)


data_input = Input(
    type="uri_folder",
    path="azureml://subscriptions/437ce2b6-c1d8-4df6-b067-fc9209c568e9/resourcegroups/Multi-policy/workspaces/multi-policy-training/datastores/workspaceblobstore/paths/UI/2026-03-19_160159_UTC/TRACE-Benchmark/LLM-CL_Benchmark/LLM-CL-Benchmark_5000"
)
model_input = Input(
    type="uri_folder",
    path="azureml://subscriptions/437ce2b6-c1d8-4df6-b067-fc9209c568e9/resourcegroups/Multi-policy/workspaces/multi-policy-training/datastores/workspaceblobstore/paths/UI/2026-03-19_160159_UTC/TRACE-Benchmark/LLM-CL_Benchmark/LLM-CL-Benchmark_5000"
)

inputs = {
  "DATA_PATH": data_input,
  "MODEL_PATH": model_input,
  "OUTPUT_DIR_PREFIX": "outputs/6datasets_CFMPSN_v1_",

  "LORA_R": 64,
  "LORA_ALPHA": 128,
  "LORA_DROPOUT": 0.0,
  "LR_CSTANCE_FOMC": 1e-5,
  "LR_MEETINGBANK": 1e-5,
  "LR_ScienceQA": 1e-5,
  "LR_Py150": 1e-5,
  "LR_NumGLUE_cm": 1e-5,

  "EPOCHS_CSTANCE": 5,
  "EPOCHS_FOMC": 3,
  "EPOCHS_MEETINGBANK": 7,
  "EPOCHS_ScienceQA": 3,
  "EPOCHS_Py150": 5,
  "EPOCHS_NumGLUE_cm": 5,

  "PER_DEVICE_BATCH_SIZE": 1,
  "GRADIENT_ACCUMULATION_STEPS": 8,
  "MAX_LENGTH": 1024,
  "WARMUP_STEPS": 30,
  "MAX_GRAD_NORM": 0.3
}
args_string = " ".join([f"--{k}={v}" for k, v in inputs.items()])
print(f"python training.py {args_string}")

job = command(
    code = "./training/", # location of source code
    command = f"python train_6datasets_CFMPSN.py {args_string}",
    environment=f"{custom_env_name}@latest",
    display_name="multi-policy-training-5000"
)

ml_client.create_or_update(job)
