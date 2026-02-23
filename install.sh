# cài môi trường
conda create -n powerpaint python=3.9 -y
conda activate powerpaint
python -m pip install -r requirements/requirements.txt
# load checkpoint
conda install git-lfs -y
git lfs install
git lfs clone https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting/ ./checkpoints/stable-diffusion-inpainting
git lfs clone https://huggingface.co/JunhaoZhuang/PowerPaint-v1/ ./checkpoints/ppt-v1