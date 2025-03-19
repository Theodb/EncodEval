# Evaluation

The evaluation module is designed for evaluating models across a variety of tasks:

- Masked Language Modeling (MLM)
- Sequence Classification (SC)
- Sequence Regression (SR)
- Token Classification (TC)
- Extractive Question Answering (EQA)
- Sentence Embedding (SE)


## Running evaluations

To run a specific evaluation, use the following command:

```bash
python main.py \ 
    --config_file <config_file_path> \ 
    --model_path <model_path>
```


## Model format

Models to be evaluated must be in HuggingFace format. If you need to convert Optimus distributed checkpoints into HuggingFace format, use the provided conversion script:

```bash
python scripts/hf_converter.py \ 
    --dcp_merger_file ../../../scripts/convert_dcp_ckpt.py \ 
    --dcp_dir <distributed_chkpt_dir> \ 
    --output_dir <hf_model_dir> \ 
    --archi_name <architecture_name> \ 
    --run_on_slurm <whether_to_run_on_slurm>
```

For more details on selecting the appropriate architecture name, refer to the conversion script located at: [src/optimus/hf_modeling/optimus_modeling/conversion.py](../hf_modeling/optimus_modeling//conversion.py).


## Task evaluation files

Files for task evaluation are located at [eval_tasks/](eval_tasks/). 


## Configuration files

Examples of configuration files are available in the [configuration/](configuration/) folder. To generate configuration files automatically from a template, use the script below:

```bash
python scripts/config_writter.py
```


## Running evaluations on SLURM

If you wish to run evaluations using SLURM, example launcher scripts are provided in the [scripts/eval_launchers/](scripts/eval_launchers/) directory.

Use example:

```bash
python scripts/eval_launchers/evaluation_launcher_sc.py \ 
    --model_dir <model_dir> \ 
    --num_gpus <num_gpus_for_slurm_job> \ 
    --wall_time <slurm_wall_time> \ 
    --run_on_slurm <whether_to_run_on_slurm>
```


## Results

Evaluation results are located in the [results/](results/) directory.