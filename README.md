# Face Privacy Preserving

1. Set the environment

```bash
conda env create -f environment.yaml
conda activate FacePrivPres
```

2. Prepare CelebA dataset

```bash
wget -O ./DATASETS/Real_CelebA_256.zip https://livewarwickac-my.sharepoint.com/:u:/g/personal/u1873231_live_warwick_ac_uk/EXdaRJf-s9BGtO_gJ3o715YBGcnLd6sNA93Vs8dtemN17Q?download=1
cd DATASETS/ && unzip Real_CelebA_256.zip && cd ..
echo "Precomputed CelebA features (Optional)"
wget -O ./DATA/CelebA_Real.csv.zip https://livewarwickac-my.sharepoint.com/:u:/g/personal/u1873231_live_warwick_ac_uk/EfPeOPhJkANGozsAeaPUtIoBv6qJb--lV7UmqXtnAFmoBg?download=1
cd DATA/ && unzip CelebA_Real.csv.zip && cd ..
```

3. Generate face traits dataset

```bash
python Main.py FaceTraits --ImagesPath ./DATASETS/CelebA/IMG/ --Labels ./DATASETS/CelebA/identity_CelebA.csv  --CropTraits 
```

4. Train Models

```bash
python Main.py TrainTraitModel --DatasetPath ./DATASETS/Traits/Eyes/ --Trait BB_Eyes --BS 3072 --Epochs 3 --LR 0.1 --Device cuda:2
```

5. ExtractFeatures

```bash
python Main.py ExtractFeatures --LandMarksFile Landmarks.csv  --ModelsPath ./MODELS/ --ImagesPath ./DATASETS/CelebA/IMG/
```

6. Train Probabilistic Model

```bash
python Main.py TrainProbabilisticModel --CSVFile ./DATA/CelebA_Real.csv
```

6. Test Probabilistic Model

```bash
python Main.py TestProbabilisticModel --SRC_CSV ./DATA/CelebA_Real.csv --TRG_CSV ./DATA/CelebA_Sample.csv --ModelsPath ./MODELS/BMM_Models_2024_06_12_10_55_44/
```
