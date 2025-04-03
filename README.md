The Adaptive Locally Weighted Random Forest (ALW-RF) algorithm enhances the traditional random forest methodology by addressing key limitations related to uniform feature treatment and computational redundancy. Unlike conventional random forest algorithms that assume equal predictive power across all features, AWL-RF assigns differential weights to attributes based on their predictive significance, calculated using Information Gain, Gain Ratio, and Pearson Correlation. This mechanism influences tree weighting during the ensemble voting process, with weights being assigned adaptively in accordance with the K nearest neighbors of the test point. The result is more efficient and accurate, and reduces the impact of irrelevant features, improves computational efficiency, and achieves higher predictive accuracy. We tested the AWL-RF algorithm with 3 different attribute selection methods in addition to the standard random forest algorithm and concluded that the AWL-RF algorithm outperformed the standard random forest algorithm in almost every case.
Traditional random forest (RF) algorithms are powerful, but suffer from two critical limitations: (1) Uniform feature treatment, in which they assume that all attributes have equal predictive power, ignoring the inherent differences in feature importance, and (2) Computational redundancy, by which they often include irrelevant attributes, increasing computational costs without improving accuracy.
To address these limitations, AWL-RF integrates feature selection directly into the forest’s structure by weighting trees based on the aggregate local predictive power of their selected attributes. This approach leverages metrics like Information Gain, Gain Ratio, and Pearson Correlation to guide both feature selection and tree weighting, aiming to reduce redundancy and enhance accuracy.


To use our code, upload a file to the same folder as split.py, run the code, and input the file name. split.py will split your file with an 80-20 train test split. Our train.csv and test.csv files that we used for testing are also provided. To run our adaptive_local_weighted_random_forest.py code on your files, upload the train.csv and test.csv files to the same folder, and run the code.
