# Datasets for training TruthLens detectors

This repository ships **no dataset**. Check each dataset's own page for its licence and
terms before downloading or training: several restrict use to non-commercial research, and
some contain people's faces. Availability and terms change, and I have not re-verified every
entry below.

## Image datasets

| Dataset | What it is | Notes |
|---|---|---|
| **GenImage** (official repo: `github.com/GenImage-Dataset/GenImage`) | Over one million pairs of AI-generated and real images. Real images use the 1,000 ImageNet classes. Generators include Midjourney, Stable Diffusion, ADM, GLIDE, Wukong, VQDM and BigGAN. | Folders are `<generator>/{train,val}/{ai,nature}`: map `ai` to `fake` and `nature` to `real`. The repository says the data is distributed through Baidu Yunpan. It defines a cross-generator task and a degraded-image task, which are the right tests for generalization. |
| A Kaggle "140k Real and Fake Faces" dataset | Face images, referenced by a Hugging Face model card | Faces only, so a model trained on it will not generalize to scenes or objects. Check its page for provenance and consent terms. |
| Your own data | Real photos you own or have permission to use, plus images from generators whose terms allow it | Best way to match your target use. Faces need a lawful basis and usually consent. |

## Video / face-swap datasets

FaceForensics++, Celeb-DF and the DFDC (Deepfake Detection Challenge) data are widely used for
video deepfakes. Access typically requires accepting terms or filling in a request form; check
each project's page. Extract frames with `training.extract_video_frames` and split by video.

## Putting data into the expected layout