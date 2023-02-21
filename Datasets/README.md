# Datasets
### Structure

Please structure your directories as below:
```
Datasets
│   README.md
│       
└───*Single Dataset* ('JSRT', 'Padchest', 'Montgomery' or 'Shenzen')
│   │   train_list.txt
│   │   val_list.txt
│   │   test_list.txt
│   │   preprocess.ipynb
│   └───Images
│   └───Landmarks
│   └───*Set* ('Train', 'Val', 'Test')
│   │   └───Images
│   │   └───Landmarks
│   │   └───Masks
│   │   └───SDF
│   
└───*Combined Dataset* ('JSRT_Padchest', or 'Montgomery_Shenzen')
│   └───*Set* ('Train', 'Val', 'Test')
│       └───Images
│       └───Landmarks
│       └───Masks
│       └───SDF
└───All_Landmarks 
│   └───H
│   └───LL
│   └───RL
```

### Data sources

Download datasets via the weblinks below and place raw images/landmarks in the corresponding folders. 

* Ground Truth: Available at [ngaggion/Chest-xray-landmark-dataset](https://github.com/ngaggion/Chest-xray-landmark-dataset), and place the `landmarks/*` directories into `All_Landmarks/` as shown above.
* JSRT Dataset: [Available here](http://db.jsrt.or.jp/eng.php)
* Padchest Dataset: [Available here](https://bimcv.cipf.es/bimcv-projects/padchest/) (Download sample 2)
* Montgomery Dataset: [Available here]("https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/CXR_png/index.html")
* Shenzen Dataset: [Available here]("https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/CXR_png/index.html")

### Preprocessing, Mask/SDF generation, and Training Splits

Run `preprocess.ipynb` in each single directory folder

### Combined Datasets

To create combined datasets, simply copy and paste files from single dataset directories