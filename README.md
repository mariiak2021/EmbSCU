# Indoor Scene Change Understanding (SCU): Segment, Describe, and Revert Any Change 
<p>Mariia Khan⋆†, Yue Qiu⋆, Yuren Cong†, Bodo Rosenhahn†, David Suter⋆†, Jumana Abu-Khalaf⋆†</p>


<p>⋆† School of Science, Edith Cowan University (ECU), Australia</p>

<p>⋆ Artificial Intelligence Research Center (AIRC), AIST, Japan</p>

<p>† Institute for Information Processing, Leibniz University of Hannover (LUH), Germany</p>

[[`Paper to be published soon`]] - accepted to [IROS24](https://www.iros2024-abudhabi.org/)

<p float="left">
  <img src="main2.JPG?raw=true" width="50%" />
  <img src="pipeline2.JPG?raw=true" width="40%" /> 
</p>

Understanding of scene changes is crucial for embodied AI applications, such as visual room rearrangement, where the agent must revert changes by restoring the objects to their original locations or states. Visual changes between two scenes, pre- and post-rearrangement, encompass two tasks: scene change detection (locating changes) and image difference captioning (describing changes). While previous methods, focused on sequential 2D images, have addressed these tasks separately, it is essential to emphasize the significance of their combination. Therefore, we propose a new Scene Change Understanding (SCU) task for simultaneous change detection and description. Moreover, we go beyond change language description generation and aim to generate rearrangement instructions for the robotic agent to revert changes. To solve this task, we propose a novel method - **EmbSCU**, which allows to compare instance-level change object masks (for 53 frequently-seen indoor object classes) before and after changes and generate rearrangement language instructions for the agent. EmbSCU is built on our **Segment Any Object Model (SAOM)** - a fine-tuned version of Segment Anything Model (SAM), adapted to obtain instance-level object masks for both foreground and background objects in indoor embodied environments. EmbSCU is evaluated on our own dataset of sequential 2D image pairs before and after changes, collected from the Ai2Thor simulator. The proposed framework achieves promising results in both change detection and change description. Moreover, EmbSCU demonstrates positive generalization results on real-world scenes without using any real-life data during training.

## News
The code for training, testing and evaluation of EmbSCU is released on 09.01.25. 

The EmbSCU datasets will be released shortly.

## 💻 Installation

To begin, clone this repository locally
```bash
git clone git@github.com:mariiak2021/EmbSCU.git 
```
<details>
<summary><b>See here for a summary of the most important files/directories in this repository</b> </summary> 
<p>

Here's a quick summary of the most important files/directories in this repository:
* `environment.yml` the file with all requirements to set up conda environment
* `pano_code`
    - `eval_results` - The folder to keep evaluation results of any model variation
    - `results` - The folder to keep checkpoints from model training
    - `MCC` - the foder with MCC-Formers training/evaluation implementation
    - `EmbSCU2` and `EmbSCU` - the foders with EmbSCU training/evaluation different implementations
    - `train.py` -  the file to train the model
    - `eval.py` - the file to evaluate the model performance
    - all other files are necessary to process EmbSCU dataset


</p>
</details>

You can then install requirements by using conda, we can create a `embclone` environment with our requirements by running
```bash
export MY_ENV_NAME=embclone
export CONDA_BASE="$(dirname $(dirname "${CONDA_EXE}"))"
export PIP_SRC="${CONDA_BASE}/envs/${MY_ENV_NAME}/pipsrc"
conda env create --file environment.yml --name $MY_ENV_NAME
```


<p>
To train the model on several GPUs run:
```bash
train.py
```

To evaluate the model run:
```bash
eval.py
```

After you get the output captions for evaluation run:
```bash
eval_accuracy.py
```

To run the re-trained model in the everything mode run:
```bash
tesbatch.py
```
</p>

