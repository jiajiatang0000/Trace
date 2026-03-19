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
    path="https://multipolicytra3370922241.blob.core.windows.net/azureml-blobstore-ac23d83f-1c41-4d32-bba0-8022c5614d4b/UI/2026-03-19_160159_UTC/TRACE-Benchmark/LLM-CL_Benchmark/LLM-CL-Benchmark_5000"
)
model_input = Input(
    type="uri_folder",
    path="azureml://subscriptions/437ce2b6-c1d8-4df6-b067-fc9209c568e9/resourcegroups/Multi-policy/workspaces/multi-policy-training/datastores/workspaceblobstore/paths/UI/2026-03-19_160159_UTC/TRACE-Benchmark/LLM-CL_Benchmark/LLM-CL-Benchmark_5000"
)

inputs = {
  "data_path": data_input,
  "model_name_or_path": model_input,
  "output_dir": "outputs/6datasets_CFMPSN_v1_",

  "lora_r": 64,
  "lora_alpha": 128,
  "lora_dropout": 0.0,
  "lr_CSTANCE_FOMC": 1e-5,
  "lr_MeetingBank": 1e-5,
  "lr_ScienceQA": 1e-5,
  "lr_Py150": 1e-5,
  "lr_NumGLUE_cm": 1e-5,

  "epochs_CSTANCE": 5,
  "epochs_FOMC": 3,
  "epochs_MeetingBank": 7,
  "epochs_ScienceQA": 3,
  "epochs_Py150": 5,
  "epochs_NumGLUE_cm": 5,

  "per_device_batch_size": 1,
  "gradient_accumulation_steps": 8,
  "max_length": 1024,
  "warmup_steps": 30,
  "max_grad_norm": 0.3
}
args_string = " ".join([f"--{k}={v}" for k, v in inputs.items()])
print(f"python training.py {args_string}")

job = command(
    code = "./training/", # location of source code
    command = f"python train_6datasets_CFMPSN.py {args_string} --bf16",
    environment=f"{custom_env_name}@latest",
    display_name="multi-policy-training-5000"
)

ml_client.create_or_update(job)
