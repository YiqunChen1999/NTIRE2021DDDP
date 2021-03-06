
# Defocus Debluring using Dual-pixel Image

## How to Start

1. create your conda environment;

2. install pytorch>=1.7.1, torchvision>=0.8.2

3. run `pip install -r requirements.txt`

4. download this repository to your local machine.

5. uncompress and orginaze the NTIRE 2021 Defocus Deblurring Test data in a folder as following:
```
DualPixelNTIRE2021:
    test:
        source:
            img_idx_0.png
            img_idx_1.png
            ...
```

6. modify 45th line in /path/to/root/of/project/src/configs/configs.py:

`"DualPixelNTIRE2021": "/path/to/your/dataset"`

for example, if you put images in: `/data/DualPixelNTIRE2021/test/source/`, then the 44th line should be: `"DualPixelNTIRE2021": "/data/DualPixelNTIRE2021/"`.

7. download the pre-trained [parameters](https://mailsdueducn-my.sharepoint.com/:u:/g/personal/201700181055_mail_sdu_edu_cn/EZqI7rKgTMRKmDpdS1Puy3kBOB3m7PgmGtO8srW0L18N5w?e=evqfii) into /path/to/root/of/project/checkpoints/Test

8. the following command should generates results in folder /path/to/root/of/project/results/Test/NTIRE2021NHHAZE/results

```bash
cd /path/to/root/of/project/
python src/main.py Test {$FolderName} 1 true [0]
```
