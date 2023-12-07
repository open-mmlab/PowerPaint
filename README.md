# PowerPaint

This README provides a step-by-step guide to download the repository, set up the required virtual environment named "PowerPaint" using conda, and run PowerPaint with or without ControlNet. **Stronger Model Weights and Online Demo Coming Soon！**

## Getting Started

### Clone the Repository

First, clone the PowerPaint repository from GitHub using the following command:

```bash
git clone git@github.com:zhuang2002/PowerPaint.git
```

### Navigate to the Repository

Enter the cloned repository directory:

```bash
cd PowerPaint
```

### Create Virtual Environment with Conda

Create and activate a virtual environment named "PowerPaint" using conda:

```bash
conda create --name PowerPaint python=3.8
conda activate PowerPaint
```

### Install Dependencies

Install the required dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Create Models Folder

Create a `models` folder within the repository directory:

```bash
mkdir models
```

### Set up Git LFS

Initialize Git LFS to manage large files efficiently:

```bash
git lfs install
```

### Clone PowerPaint Model

Clone the PowerPaint model using Git LFS:

```bash
git lfs clone https://huggingface.co/JunhaoZhuang/PowerPaint-v1/ ./models
```

## Run PowerPaint

To run PowerPaint, execute the following command:

```bash
python gradio_PowerPaint.py
```

This command will launch the Gradio interface for PowerPaint.

## Using PowerPaint with ControlNet

PowerPaint can be used in conjunction with ControlNet. The project supports ControlNet integration for human pose, HED, Canny, and depth. To use PowerPaint with ControlNet, execute the following command:

```bash
python gradio_PowerPaint_Controlnet.py
```

This command will launch the Gradio interface for PowerPaint with ControlNet.

Feel free to explore and create stunning images with PowerPaint!

## Contributors

## License
