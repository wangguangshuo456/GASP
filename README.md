
# Directory Structure

  We use the HealthCareMagic-100k, LLaMA and MiniLM scenario as an example to illustrate the directory structure.

```
  ├── README.md

  ├── requirements.txt
  
  ├─ DatasetsRaw/         

  │   ├─ HealthCareMagic/   

  │   ├─ AgNews/

  │   ├─ NQ/  

  │   └─ MsMarco/           
  
  ├── EmbeddingModel
  
  │   ├── sentence-transformers
  
  │       ├── all_minilm_l6_v2

  ├── LLM

  │   ├── meta-llama

  │       ├── meta-llama-3-8b-instruct
  
  ├── utils

  │   ├── file_read_utils.py

  │   ├── llms.py

  │   ├── roc_plot.py

  │   ├── similarity_get.py

  │   └─ vector_db_utils.py 
  
  ├── computeBLEUs.py

  ├── computeBLEUsShadow.py
  
  ├── computeEditDistance.py

  ├── computeEditDistanceShadow.py

  ├── computeRouge.py
  
  ├── computeRougeShadow.py

  ├── data_process.py

  ├── FuzzCluster.py
  
  ├── Models.py

  ├── QueryShadowRAG.py

  ├── QueryTargetRAG.py

  ├── RAG_shadow_create.py

  ├── RAG_target_create.py

  ├── trainAttackModel.py

  └── README.md
```
  
# Supported Dataset

  HealthCareMagic-100k, MS-MARCO, Natural Questions, AGNews

# Supported LLM

  Meta-Llama-3-8B-Instruct, Mistral-7B-Instruct-v0.2, GLM-4-9B-Chat
  
# Environment Dependencies
  Create a new Python environment using Conda, specifying the environment name and Python version: `conda create -n env_name python=3.8`.

  Once the environment is created, activate it and use pip to install all the packages listed in the `requirements.txt` file. To do this, first activate the environment with: `conda activate env_name`, then run the following command to install the dependencies: `pip install -r requirements.txt`.

  Alternatively, you can manually install the dependencies listed in the `requirements.txt` file.  

  Finally, switch the project environment to the newly created one in order to run the project.

  Please note that the hardware requirements are as follows: a minimum configuration of an NVIDIA RTX A6000 is required. Additionally, the attack was tested on Windows and has not yet been evaluated on Linux.

# Usage Instructions

  ## Step 0. Preprocess the dataset
  For HealthCareMagic-100k, please place the HealthCareMagic-100k dataset file `HealthCareMagic-100k-en.jsonl`(downloaded from the website https://huggingface.co/datasets/RafaelMPereira/HealthCareMagic-100k-Chat-Format-en/tree/main) into the `.\DatasetsRaw\HealthCareMagic` folder. 

  For MS-MARCO, please place the MS-MARCO dataset files (downloaded from the website https://huggingface.co/datasets/microsoft/ms_marco/tree/main/v2.1, including `train-00000-of-00007.parquet` through `train-00006-of-00007.parquet` files) into the the `.\DatasetsRaw\MsMarco` folder. 

  For Natural Questions, please place the Natural Questions dataset files (downloaded from the website https://huggingface.co/datasets/addy88/nq-question-answeronly/tree/main/data, including `train-00000-of-00024.parquet` through `train-00023-of-00024.parquet` files) into the the `.\DatasetsRaw\NQ` folder. 
  
  For AGNews, please place the AGNews dataset file `train-00000-of-00001.parquet` (downloaded from the website https://huggingface.co/datasets/fancyzhx/ag_news/tree/main/data) into the the `.\DatasetsRaw\AgNews` folder. 
  
  Then, you can run `python data_process.py` to preprocess these datasets. Before running the script, set the `datasetName` variable in the main function to the desired dataset name (valid options are `HealthCareMagic`, `NQ`, `MsMarco`, or `AGNews`). The processed data will be saved in the `Datasets` directory.

  ## Step 1. Configure the retriever and generator for RAG.
  The setup process is illustrated using LLaMA and MiniLM.

  `MiniLM`: Download `all-MiniLM-L6-v2` from https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 and place it in the directory `./EmbeddingModel/sentence-transformers/all_minilm_l6_v2`.
  
  `LLaMA`: Download `Meta-Llama-3-8B-Instruct` from https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/tree/main, which will serve as the RAG generator, and place it in the `./LLM` directory. For instance, the path could be `./LLM/meta-llama/meta-llama-3-8b-instruct`.

  ## Step 2. Create the Target RAG and Shadow RAG.
  Create the Target RAG and Shadow RAG by running `RAG_target_create.py` and `RAG_shadow_create.py`, respectively. Before executing the scripts, set the `datasetName` variable to specify the dataset to be used for RAG creation (valid options are `HealthCareMagic`, `NQ`, `MsMarco`, or `AGNews`). The RAG knowledge base and the partitioned datasets will be stored in the `VectorDatabase` and `SamplesDatabase` directories, respectively.

  ## Step 3. Query the Target and Shadow RAG Systems.
  Execute `QueryTargetRAG.py` and `QueryShadowRAG.py` separately to query the Target RAG and Shadow RAG systems, thereby obtaining the raw output data (similarity scores are also computed during this process). The results are saved under the corresponding subdirectories within the `AttackResult` directory. For example, the outputs may be stored in paths such as `./AttackResult/HealthCareMagic/meta_CSA_attack/PredictValues_Seq` and `PredictValues_Seq_Shadow`.

  Note: Before running the scripts, ensure that the following variables are properly configured:

  `llm_name`: the LLM provider (options: `meta`, `mistralai`, `chatglm`)

  `datasets_name`: the dataset used (options: `HealthCareMagic`, `NQ`, `MsMarco`, `AGNews`)

  Additionally, `max_tokens_list` specifies the generation budget used in our approach to construct the side channel.

  ## Step 4. Compute Metrics for Generation Quality.
  In addition to similarity computation, several other metrics are calculated using the following scripts.

  Run `computeEditDistance.py` and `computeEditDistanceShadow.py` to compute edit distance.

  Run `computeRouge.py` and `computeRougeShadow.py` to calculate ROUGE scores (e.g., ROUGE-L, ROUGE-1, ROUGE-2).

  Run `computeBLEUs.py` and `computeBLEUsShadow.py` to compute BLEU scores.

  All resulting metric values are saved in their respective subdirectories under `./AttackResult/`.

  Note: The `max_tokens_list` parameter, which defines the generation budget used in our approach to construct the side channel, must remain consistent with the value used in the previous steps.

  ## Step 5. Perform the GASP-S Attack.
  Execute `trainAttackModel.py` to perform the GASP-S membership inference attack. The following parameters must be specified:

  `llm_name` and `datasets_name`: These should remain consistent with previous steps, indicating the target LLM (e.g., meta, mistralai) and dataset (e.g., HealthCareMagic, NQ), respectively.

  `token_nums`: Specifies the generation budget used in our side channel construction, which must match the max_tokens_list value used in prior steps.

  `metrics`: A list of evaluation metrics used to train the attack model, such as: `["similarity", "rouge1", "rouge2", "rougeL", "rougeLsum", "edit_distance", "BLEU4"]`. This list can be customized, but all specified metrics must have been computed in the previous step, otherwise the script will raise an error.

  `attack_epoch`: The number of training epochs, typically ranging from 200 to 500, with optimal values depending on the dataset and requiring tuning.
  
  The attack performance will be printed, including `Accuracy`, `AUC`, and `TPR at 0.001 FPR`.

  ## Step 6. Perform the GASP-U Attack.

  Execute `FuzzCluster.py` to perform the GASP-U membership inference attack. The following parameters must be specified:

  `llm_name` and `datasets_name`: These should remain consistent with previous steps, indicating the target LLM (e.g., meta, mistralai) and dataset (e.g., HealthCareMagic, NQ), respectively.

  `token_nums`: Specifies the generation budget used in our side channel construction, which must match the max_tokens_list value used in prior steps.

  `metrics`: A list of evaluation metrics used to train the attack model, such as: `["similarity", "rouge1", "rouge2", "rougeL", "rougeLsum", "edit_distance", "BLEU4"]`. This list can be customized, but all specified metrics must have been computed in the previous step, otherwise the script will raise an error.

  The attack performance will be printed, including `Accuracy`, `AUC`, and `TPR at 0.001 FPR`.

  Note: The entire experiment takes approximately 50 hours to complete, primarily due to the time required to collect side-channel signals for training the attack model. However, once the model is trained, inferring the membership status of a new sample takes less than one minute.



  



  
